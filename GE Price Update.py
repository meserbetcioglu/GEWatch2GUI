def request(url, params = None, tag = None, attrs = None):
    import requests as req
    try:
        from BeautifulSoup import BeautifulSoup
    except ImportError:
        from bs4 import BeautifulSoup

    headers = {
        'User-Agent': 'mserbet#6975', #Enter your discord handle
        'From': 'meserbetcioglu@gmail.com' #Enter your gmail
        }

    response = req.get(url, headers = headers, params = params).text

    return response

def first_update(dir):

  dt = datetime.datetime.strptime(input('Enter the first date of pricing (DD-MM-YYYY HH): '), '%d-%m-%Y %H') + datetime.timedelta(hours=3)

  logger.info('============ First update starting ============')
  logger.info(f'Timestamp: {datetime.datetime.timestamp(dt)}')


  bulk_prices_history = {}

  try:
      h_list = json.loads(request('https://prices.runescape.wiki/api/v1/osrs/1h', {'timestamp' : int(datetime.datetime.timestamp(dt))}))
  except Exception as e: # Simplified try-except
      logger.error(f"Error fetching initial data: {e}")
      return {}

  h_list['timestamp'] = datetime.datetime.utcfromtimestamp(h_list['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
  bulk_prices_history[h_list['timestamp']] = h_list

  try:
      _atomic_json_dump(dir, bulk_prices_history)
  except Exception as e: # Simplified try-except, removed redundant file.close()
      logger.error(f"Error saving initial price history: {e}")

  logger.info(f"Loaded: {h_list['timestamp']}")

  return bulk_prices_history


def remove_past_price(dir):

    logger.info('============ Remove past price starting ============')

    # Keep a wider retention window so charting can zoom out well beyond 1 month.
    retention_days = 90
    cutoff_dt = datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0) - datetime.timedelta(days=retention_days)

    bulk_prices_history = _load_price_history_with_recovery(dir)
    if not bulk_prices_history:
        return {}

    past_prices_history = {key:bulk_prices_history[key] for key in bulk_prices_history if datetime.datetime.strptime(key, '%Y-%m-%d %H:%M:%S') < cutoff_dt}

    if len(past_prices_history.keys()) > 0:
        logger.info(f'Earliest found invalid date: {min(past_prices_history.keys())}')
        logger.info(f'Latest found invalid date: {max(past_prices_history.keys())}')
    elif len(bulk_prices_history.keys()) > 0:
        logger.info('Past price history not found.')
        return bulk_prices_history
    else:
        logger.warning('Price history not found.')
        return {}

    bulk_prices_history = {key:bulk_prices_history[key] for key in bulk_prices_history if datetime.datetime.strptime(key, '%Y-%m-%d %H:%M:%S') >= cutoff_dt}

    logger.info(f'History cleaned, {min(bulk_prices_history.keys())} to {max(bulk_prices_history.keys())} is left.')

    return bulk_prices_history


def backfill_min_history(dir, bulk_prices_history, min_days=90):

    if not bulk_prices_history:
        return bulk_prices_history

    logger.info(f'============ Backfill history to at least {min_days} days ============')

    target_oldest = datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0) - datetime.timedelta(days=min_days)
    save_every = 24
    fetched = 0

    while True:
        try:
            oldest_str = min(bulk_prices_history.keys())
            oldest_dt = datetime.datetime.strptime(oldest_str, '%Y-%m-%d %H:%M:%S')
        except Exception as e:
            logger.error(f'Could not parse oldest timestamp for backfill: {e}')
            break

        if oldest_dt <= target_oldest:
            break

        requested_dt = oldest_dt - datetime.timedelta(hours=1)
        requested_ts = int(requested_dt.replace(tzinfo=datetime.timezone.utc).timestamp())

        try:
            h_list = json.loads(request('https://prices.runescape.wiki/api/v1/osrs/1h', params={'timestamp': requested_ts}))
            returned_ts = datetime.datetime.utcfromtimestamp(h_list['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            if returned_ts in bulk_prices_history:
                logger.warning(f'Backfill returned duplicate timestamp {returned_ts}; stopping backfill to avoid infinite loop.')
                break
            h_list['timestamp'] = returned_ts
            bulk_prices_history[returned_ts] = h_list
            fetched += 1
            logger.info(f'Backfilled: {returned_ts}')
        except Exception as e:
            logger.error(f'Backfill request failed for {requested_dt}: {e}')
            break

        if fetched % save_every == 0:
            try:
                _atomic_json_dump(dir, bulk_prices_history)
            except Exception as e:
                logger.error(f'Error saving during backfill: {e}')

        # Small pause to stay friendly to the API while backfilling.
        time.sleep(0.25)

    if fetched > 0:
        try:
            _atomic_json_dump(dir, bulk_prices_history)
        except Exception as e:
            logger.error(f'Error saving final backfill data: {e}')

    logger.info(f'Backfill complete. Added {fetched} hourly points.')
    return bulk_prices_history


def bootstrap_history_from_latest_hour(dir):
    logger.info('============ Bootstrap history from latest hour ============')
    target_dt = datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0) - datetime.timedelta(hours=1)
    target_timestamp = int(target_dt.replace(tzinfo=datetime.timezone.utc).timestamp())

    try:
        h_list = json.loads(request('https://prices.runescape.wiki/api/v1/osrs/1h', params={'timestamp': target_timestamp}))
        h_list['timestamp'] = datetime.datetime.utcfromtimestamp(h_list['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        bulk_prices_history = {h_list['timestamp']: h_list}
        _atomic_json_dump(dir, bulk_prices_history)
        logger.info(f"Bootstrapped history with: {h_list['timestamp']}")
        return bulk_prices_history
    except Exception as e:
        logger.error(f'Bootstrap failed: {e}')
        return {}

def update_to_latest_price(dir, bulk_prices_history, last_time = None):

    if bulk_prices_history == None or len(bulk_prices_history) == 0:
        logger.warning('No usable history found; bootstrapping from latest available hour.')
        bulk_prices_history = bootstrap_history_from_latest_hour(dir)
        if not bulk_prices_history:
            logger.error('Cannot continue update_to_latest_price without bootstrap data.')
            return

    # Ensure minimum history depth so GUI charts can show long-window trends.
    bulk_prices_history = backfill_min_history(dir, bulk_prices_history, min_days=90)


    logger.info('============ Update to latest price starting ============')

    i = 1
    while(True):

        dt = datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0) - datetime.timedelta(hours = 1)
        logger.info(f'Update to latest price iteration: {i}')
        
        # Debug logging
        logger.debug(f'Current dt (target hour - 1): {dt}')
        #print('Current Time: ', dt)
        #print('Timestamp: ', datetime.datetime.timestamp(dt))
        #print('Timestamp UTC: ', datetime.datetime.timestamp(dt - datetime.timedelta(hours = 3)))

        if last_time == None:
            last_time = max(bulk_prices_history.keys(), default = (datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0) - datetime.timedelta(days = 90)).strftime('%Y-%m-%d %H:%M:%S'))
        else:
            last_time = (datetime.datetime.strptime(last_time, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours = 1)).strftime('%Y-%m-%d %H:%M:%S')

        logger.debug(f'last_time after increment: {last_time}')
        logger.debug(f'Checking if {dt} > {last_time}')
        
        if datetime.datetime.timestamp(dt) > datetime.datetime.timestamp(datetime.datetime.strptime(last_time, '%Y-%m-%d %H:%M:%S')):
            try: # Simplified try-except
                # Parse last_time as UTC and convert to Unix timestamp correctly
                next_hour_utc = datetime.datetime.strptime(last_time, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=1)
                next_hour_timestamp = int(next_hour_utc.replace(tzinfo=datetime.timezone.utc).timestamp())
                logger.debug(f'Requesting data for {next_hour_utc} (timestamp: {next_hour_timestamp})')
                h_list = json.loads(request('https://prices.runescape.wiki/api/v1/osrs/1h', params = {'timestamp': next_hour_timestamp}))
            except Exception as e:
                logger.error(f"Error fetching hourly data for {last_time}: {e}")
                logger.info('Retrying in 60 seconds...')
                time.sleep(60)
                continue # Retry instead of giving up

            h_list['timestamp'] = datetime.datetime.utcfromtimestamp(h_list['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            bulk_prices_history[h_list['timestamp']] = h_list
        else:
            logger.info(f'Data is up to date. dt: {last_time}')
            break

        logger.info(f'Loaded: {h_list["timestamp"]}')

        if i%24 == 0:
            logger.info('Starting to save, do not close.')
            now = datetime.datetime.timestamp(datetime.datetime.now())
            try:
                _atomic_json_dump(dir, bulk_prices_history)
            except Exception as e: # Simplified try-except, removed redundant file.close()
                logger.error(f"Error saving price history: {e}")
            logger.info(f'Finished saving. Took {int(datetime.datetime.timestamp(datetime.datetime.now()) - now)} seconds.')

        i += 1
        time.sleep(10)


    if i != 1:
        logger.info('Starting the final save, do not close.')
        now = datetime.datetime.timestamp(datetime.datetime.now())
        try:
            _atomic_json_dump(dir, bulk_prices_history)
        except Exception as e: # Simplified try-except, removed redundant file.close()
            logger.error(f"Error saving final price history: {e}")
        logger.info(f'Finished saving. Took {int(datetime.datetime.timestamp(datetime.datetime.now()) - now)} seconds.')

def update_price(dir):

    logger.info('============ Update price starting ============')

    bulk_prices_history = _load_price_history_with_recovery(dir)
    if not bulk_prices_history:
        logger.warning('Price history unavailable after recovery attempts; trying bootstrap.')
        bulk_prices_history = bootstrap_history_from_latest_hour(dir)
        if not bulk_prices_history:
            logger.error('Price history is still unavailable after bootstrap.')
            return

    i = 1
    while(True):

        dt = datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        target_timestamp_str = (dt - datetime.timedelta(hours = 1)).strftime('%Y-%m-%d %H:%M:%S')

        # Check what's the latest data we have
        if bulk_prices_history:
            latest_in_file = max(bulk_prices_history.keys())
            logger.debug(f'Latest data in file: {latest_in_file}, Target: {target_timestamp_str}')
        else:
            latest_in_file = None
            logger.warning('No data in history file!')

        if target_timestamp_str not in bulk_prices_history:
            logger.info(f'Update price iteration: {i}')
            try:
                # Calculate UTC timestamp correctly
                target_hour_utc = dt - datetime.timedelta(hours = 1)
                target_timestamp = int(target_hour_utc.replace(tzinfo=datetime.timezone.utc).timestamp())
                logger.debug(f'Requesting data for {target_hour_utc} (timestamp: {target_timestamp})')
                h_list = json.loads(request('https://prices.runescape.wiki/api/v1/osrs/1h', params = {'timestamp': target_timestamp}))
            except Exception as e:
                logger.error(f"Error fetching hourly data for {target_timestamp_str}: {e}")
                # Decide how to handle: break, continue, retry
                time.sleep(60) # Wait a minute before retrying or breaking
                continue

            h_list['timestamp'] = datetime.datetime.utcfromtimestamp(h_list['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            
            # Check if we got the data we requested
            if h_list['timestamp'] != target_timestamp_str:
                logger.warning(f"API returned {h_list['timestamp']} but we requested {target_timestamp_str}. This may indicate a timezone bug or API issue.")
            
            bulk_prices_history[h_list['timestamp']] = h_list

            logger.info(f'Loaded: {h_list["timestamp"]}')

            i += 1

            # Save after fetching new data
            logger.info('Starting save, do not close.')
            now = datetime.datetime.timestamp(datetime.datetime.now())
            try:
                _atomic_json_dump(dir, bulk_prices_history)
            except Exception as e:
                logger.error(f"Error saving price history: {e}")
            logger.info(f'Finished saving. Took {int(datetime.datetime.timestamp(datetime.datetime.now()) - now)} seconds.')
        else:
            logger.info(f'Data for {target_timestamp_str} already exists.')

        # Wait for next hour boundary
        current_time = datetime.datetime.utcnow()
        # Calculate when 5 minutes past the next hour is
        next_hour_start = (current_time + datetime.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        wait_until = next_hour_start + datetime.timedelta(minutes=5) # Wait until 5 minutes past the next hour
        
        logger.info(f'Waiting until {wait_until.strftime("%H:%M:%S")} UTC for next update...')
        
        # Sleep until next update time (check every 30 seconds to avoid log spam)
        while datetime.datetime.utcnow() < wait_until:
            time.sleep(30)


import json
import time
import os
import sys
import datetime
import logging
import argparse
import shutil
# from google.colab import drive
# drive.mount('/content/drive')

# dir_path = '/content/drive/MyDrive/Colab/GEWatch/content' #Price history file location. This is set for drive, you may want to change it.
dir_path = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
data_file = 'Price_History.json' #Try to save back ups from time to time. If the connection cuts off midsave file becomes corrupted.
data_dir = os.path.join(dir_path, 'Data', data_file)

# Setup logging
log_file = os.path.join(dir_path, 'price_updater_status.log')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Unbuffer stdout for real-time console output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None


def _price_history_backup_path(path):
    return f"{path}.bak"


def _atomic_json_dump(path, payload):
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file)
        file.flush()
        os.fsync(file.fileno())

    try:
        if os.path.exists(path):
            try:
                shutil.copy2(path, _price_history_backup_path(path))
            except Exception as e:
                logger.warning(f"Could not refresh backup file before replace: {e}")
        os.replace(tmp_path, path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise


def _load_price_history_with_recovery(path):
    backup_path = _price_history_backup_path(path)

    try:
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        logger.warning('Price history file not found.')
        return {}
    except json.JSONDecodeError as e:
        logger.error(f'Error decoding primary price history JSON: {e}')
    except Exception as e:
        logger.error(f'Unexpected error while loading primary price history: {e}')

    if not os.path.exists(backup_path):
        logger.error('No backup price history file available for recovery.')
        return {}

    try:
        with open(backup_path, 'r', encoding='utf-8') as file:
            backup_data = json.load(file)
        if not isinstance(backup_data, dict):
            logger.error('Backup price history file is not a JSON object.')
            return {}
        logger.warning('Recovered price history from backup file.')
        try:
            _atomic_json_dump(path, backup_data)
        except Exception as restore_err:
            logger.warning(f'Could not restore primary file from backup: {restore_err}')
        return backup_data
    except Exception as e:
        logger.error(f'Failed to load backup price history file: {e}')
        return {}





if not os.path.exists(os.path.join(dir_path, 'Data')):
    os.makedirs(os.path.join(dir_path, 'Data'))




def main():
    parser = argparse.ArgumentParser(description='OSRS price history updater')
    parser.add_argument('--backfill-only', action='store_true', help='Backfill history and exit without starting hourly loop')
    parser.add_argument('--min-days', type=int, default=90, help='Minimum history days to backfill when --backfill-only is used')
    args = parser.parse_args()

    cleaned_history = remove_past_price(data_dir)

    if args.backfill_only:
        backfill_min_history(data_dir, cleaned_history, min_days=max(1, args.min_days))
        logger.info('Backfill-only mode complete.')
        return

    update_to_latest_price(data_dir, cleaned_history)
    update_price(data_dir)


if __name__ == '__main__':
    main()
