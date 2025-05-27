import re
import asyncio
import sys
from telethon import TelegramClient
from telethon.sessions import StringSession # Import StringSession
from telethon.errors.rpcerrorlist import ChannelPrivateError, MessageNotModifiedError, FloodWaitError, PeerIdInvalidError, UserNotParticipantError
from telethon.tl.types import Channel, User, Chat
from telethon.tl.functions.channels import GetMessagesRequest
from urllib.parse import urlparse
from tqdm.asyncio import tqdm as async_tqdm

from config import TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_SESSION_NAME, TELEGRAM_SESSION_STRING

# Initialize the client globally.
if TELEGRAM_SESSION_STRING:
    telethon_client = TelegramClient(StringSession(TELEGRAM_SESSION_STRING), TELEGRAM_API_ID, TELEGRAM_API_HASH)
else:
    telethon_client = TelegramClient(TELEGRAM_SESSION_NAME, TELEGRAM_API_ID, TELEGRAM_API_HASH)


async def connect_telethon_client():
    """Connects the global Telethon client. Should be called once on app startup."""
    if not telethon_client.is_connected():
        print("Connecting Telethon client...", file=sys.stderr)
        try:
            await telethon_client.start()
            if not await telethon_client.is_user_authorized(): # Awaited
                print("WARNING: Telethon client is not authorized. Scraping might fail.", file=sys.stderr)
            print("Telethon client connected successfully.", file=sys.stderr)
        except ConnectionRefusedError:
            print("ERROR: Telegram authorization failed. Check API ID/HASH or Session String.", file=sys.stderr)
            raise
        except Exception as e:
            print(f"ERROR: Failed to connect Telethon client: {e}", file=sys.stderr)
            raise
    else:
        print("Telethon client already connected.", file=sys.stderr)


async def disconnect_telethon_client():
    """Disconnects the global Telethon client. Should be called on app shutdown."""
    if telethon_client.is_connected():
        print("Disconnecting Telethon client...", file=sys.stderr)
        await telethon_client.disconnect()
        print("Telethon client disconnected.", file=sys.stderr)


async def parse_telegram_url(url):
    parsed_url = urlparse(url)
    path_parts = [p for p in parsed_url.path.split('/') if p]

    if not path_parts:
        return {'type': 'invalid'}

    telegram_prefixes = ['t.me', 'telegram.me']
    if not any(prefix in parsed_url.netloc for prefix in telegram_prefixes):
        return {'type': 'invalid'}

    url_info = {'original_url': url}

    if len(path_parts) == 1:
        url_info.update({'type': 'channel_or_group', 'identifier': path_parts[0]})
        return url_info
    elif len(path_parts) == 2 and path_parts[0] == 'c':
        try:
            url_info.update({'type': 'channel_or_group', 'identifier': int(path_parts[1])})
            return url_info
        except ValueError:
            return {'type': 'invalid'}

    if len(path_parts) >= 2:
        try:
            message_id = int(path_parts[-1])
            channel_identifier = path_parts[-2]
            if channel_identifier == 'c':
                 channel_identifier = int(path_parts[-2])
            
            url_info.update({'type': 'message', 'identifier': channel_identifier, 'message_id': message_id})
            return url_info
        except ValueError:
            return {'type': 'invalid'}
    
    return {'type': 'invalid'}


async def _get_entity(identifier):
    """
    Helper to get Telethon entity (Channel or Chat) using the already connected client.
    Ensures 'entity' is always defined to prevent UnboundLocalError.
    """
    entity = None # <--- Crucial: Initialize entity to None
    try:
        entity = await telethon_client.get_entity(identifier)
        if isinstance(entity, User):
            raise ValueError(f"Error: '{identifier}' is a user, not a channel or group. Analysis is for channels/groups.", file=sys.stderr)
        if not isinstance(entity, (Channel, Chat)):
             raise ValueError(f"Error: Unknown entity type for '{identifier}'. Must be a channel or group.", file=sys.stderr)
        
        return entity
    except PeerIdInvalidError:
        raise ValueError(f"Error: Channel/group '{identifier}' not found or invalid ID. Please check the URL.", file=sys.stderr)
    except UserNotParticipantError:
        raise ValueError(f"Error: You are not a participant in channel/group '{identifier}'. Cannot fetch content.", file=sys.stderr)
    except Exception as e:
        # Re-raise the original exception to be caught higher up
        raise RuntimeError(f"An unexpected error occurred while getting channel/group entity for '{identifier}': {e}") from e


async def get_telegram_comments_for_message(channel_identifier, message_id):
    comments = []
    try:
        entity = await _get_entity(channel_identifier)
        # If _get_entity raised an error, this line would not be reached,
        # so `entity` is guaranteed to be a valid entity object here.

        target_message = None
        try:
            messages_response = await telethon_client(GetMessagesRequest(peer=entity, id=[message_id]))
            if messages_response.messages:
                target_message = messages_response.messages[0]
            if not target_message or target_message.id != message_id:
                print(f"Warning: Message ID {message_id} not found in {entity.title}.", file=sys.stderr)
                return []
        except Exception as e:
            print(f"Error fetching target message {message_id} from {entity.title}: {e}", file=sys.stderr)
            return []

        if isinstance(entity, Channel) and hasattr(entity, 'linked_chat_id') and entity.linked_chat_id:
            try:
                discussion_group = await telethon_client.get_entity(entity.linked_chat_id)
                async for msg in telethon_client.iter_messages(discussion_group, reply_to=target_message.id):
                    if msg.text:
                        comments.append(msg.text)
            except FloodWaitError as e:
                print(f"Telegram flood wait: Waiting {e.seconds}s.", file=sys.stderr)
                await asyncio.sleep(e.seconds + 1)
            except Exception as e:
                pass 
        
        if not comments: # Only if comments haven't been found via linked chat
            try:
                async for msg in telethon_client.iter_messages(entity, reply_to=message_id):
                    if msg.text:
                        comments.append(msg.text)
            except FloodWaitError as e:
                print(f"Telegram flood wait: Waiting {e.seconds}s.", file=sys.stderr)
                await asyncio.sleep(e.seconds + 1)
            except Exception as e:
                pass

    except FloodWaitError as e:
        print(f"Telegram flood wait: Waiting for {e.seconds} seconds before retrying comment scrape...", file=sys.stderr)
        await asyncio.sleep(e.seconds + 1)
        return await get_telegram_comments_for_message(channel_identifier, message_id)
    except Exception as e: # Catch any exceptions propagated from _get_entity or other issues
        print(f"An error occurred while fetching comments for message: {e}", file=sys.stderr)
        return []
    finally:
        pass 
    return comments


async def get_channel_or_group_content(identifier, message_limit=None):
    channel_content_data = []
    messages_with_comments_count = 0
    total_comments_retrieved = 0

    try:
        entity = await _get_entity(identifier)
        # If _get_entity raised an error, this line would not be reached,
        # so `entity` is guaranteed to be a valid entity object here.

        entity_type_name = type(entity).__name__
        messages_fetched_count = 0
        messages_iter = telethon_client.iter_messages(entity, limit=message_limit)

        pbar = async_tqdm(total=message_limit if message_limit else None, desc=f"Scraping messages from {entity.title}", unit="msg")
        
        while True:
            try:
                message = await asyncio.wait_for(anext(messages_iter), timeout=30.0)
                pbar.update(1)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                print(f"Timeout while fetching messages from {entity.title}. Stopping scrape early.", file=sys.stderr)
                break
            except FloodWaitError as e:
                print(f"FloodWait during main channel/group scrape: Waiting for {e.seconds} seconds...", file=sys.stderr)
                await asyncio.sleep(e.seconds + 1)
                continue
            except Exception as e:
                print(f"Error iterating messages from {entity.title}: {e}", file=sys.stderr)
                break

            if message.text and message.id:
                message_data = {'message_id': message.id, 'message_text': message.text, 'comments': []}
                
                comments_for_message = []
                if isinstance(entity, Channel) and hasattr(entity, 'linked_chat_id') and entity.linked_chat_id:
                    try:
                        discussion_group = await telethon_client.get_entity(entity.linked_chat_id)
                        async for comment_msg in telethon_client.iter_messages(discussion_group, reply_to=message.id):
                            if comment_msg.text:
                                comments_for_message.append(comment_msg.text)
                    except FloodWaitError as e:
                        print(f"FloodWait during channel comment scrape: Waiting {e.seconds}s.", file=sys.stderr)
                        await asyncio.sleep(e.seconds + 1)
                    except Exception as e:
                        pass
                
                if not comments_for_message:
                    try:
                        async for comment_msg in telethon_client.iter_messages(entity, reply_to=message.id):
                            if comment_msg.text:
                                comments_for_message.append(comment_msg.text)
                    except FloodWaitError as e:
                        print(f"FloodWait during group comment scrape: Waiting {e.seconds}s.", file=sys.stderr)
                        await asyncio.sleep(e.seconds + 1)
                    except Exception as e:
                        pass

                if comments_for_message:
                    messages_with_comments_count += 1
                    total_comments_retrieved += len(comments_for_message)
                    message_data['comments'] = comments_for_message

                channel_content_data.append(message_data)
                messages_fetched_count += 1
            
            if message_limit and messages_fetched_count >= message_limit:
                break
        
        pbar.close()

    except FloodWaitError as e:
        print(f"Telegram flood wait during main channel/group scrape: Waiting for {e.seconds} seconds...", file=sys.stderr)
        await asyncio.sleep(e.seconds + 1)
    except Exception as e: # Catch any exceptions propagated from _get_entity or other issues
        print(f"An error occurred during channel/group content scraping: {e}", file=sys.stderr)
    finally:
        pass 
    
    print(f"Finished scraping. Retrieved {len(channel_content_data)} messages from {entity.title}. {messages_with_comments_count} messages had comments.", file=sys.stderr)
    return channel_content_data, messages_with_comments_count, total_comments_retrieved


# Example usage for testing this module independently:
async def main_scraper_test():
    print("--- Telegram Scraper Test (Requires Authorization) ---")
    print("This script helps you authorize your Telethon session and test scraping.")
    print("Enter 'exit' to quit.")
    print("URLs can be for a channel/group (e.g., https://t.me/channel_username) or a specific post (https://t.me/channel_username/message_id).")
    await connect_telethon_client()
    try:
        while True:
            test_url = input("\nEnter a Telegram URL: ")
            if test_url.lower() == 'exit':
                break

            parsed_info = await parse_telegram_url(test_url)

            if parsed_info['type'] == 'channel_or_group':
                print(f"Parsed: Channel/Group Identifier: {parsed_info['identifier']}")
                channel_content, msgs_with_comments, total_cmnts = await get_channel_or_group_content(parsed_info['identifier'], message_limit=5)
                print(f"\nRetrieved {len(channel_content)} messages from channel/group. {msgs_with_comments} had comments ({total_cmnts} total comments).")
                for i, item in enumerate(channel_content):
                    print(f"--- Message {i+1} (ID: {item['message_id']}) ---")
                    print(f"Message: {item['message_text'][:100]}...")
                    if item['comments']:
                        print(f"Comments ({len(item['comments'])}): {item['comments'][0][:100]}...")
                    else:
                        print("No comments.")
            elif parsed_info['type'] == 'message':
                print(f"Parsed: Channel/Group ID: {parsed_info['identifier']}, Message ID: {parsed_info['message_id']}")
                comments = await get_telegram_comments_for_message(parsed_info['identifier'], parsed_info['message_id'])
                print(f"\nRetrieved {len(comments)} comments.")
                for i, comment in enumerate(comments[:5]):
                    print(f"{i+1}. {comment[:100]}...")
                if not comments:
                    print("No comments found or error occurred during scraping.")
            else:
                print("Invalid Telegram URL provided. Please use a valid channel/group or post URL.")
    finally:
        await disconnect_telethon_client()

if __name__ == "__main__":
    asyncio.run(main_scraper_test())