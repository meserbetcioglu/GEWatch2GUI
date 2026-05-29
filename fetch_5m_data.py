import requests
import os
import sys
import json
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

APP_DIR = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, 'Data')
FIVE_M_FILE = os.path.join(DATA_DIR, '5m_data_cache.json')
API_URL = 'https://prices.runescape.wiki/api/v1/osrs/5m'
HEADERS = {
    'User-Agent': 'mserbet#6975',  # Enter your discord handle
    'From': 'meserbetcioglu@gmail.com'  # Enter your gmail
}

def get_5m_chunk_timestamp(dt):
    # Returns the timestamp (UTC) for the start of the 5m chunk containing dt
    dt = dt.replace(second=0, microsecond=0)
    minute = (dt.minute // 5) * 5
    return dt.replace(minute=minute)

def fetch_5m_chunk(ts):
    # ts: datetime object (UTC)
    ts_epoch = int(ts.timestamp())
    url = f'{API_URL}?timestamp={ts_epoch}'
    print(f'Fetching 5m chunk for {ts_epoch}...')
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    # Save timestamp in UTC
    return ts.strftime('%Y-%m-%dT%H:%M:%S'), response.json()

def fetch_and_save_5m_data():
    # Load existing cache and reuse any chunks that still belong in the active window.
    if os.path.exists(FIVE_M_FILE):
        with open(FIVE_M_FILE, 'r', encoding='utf-8') as f:
            try:
                cache = json.load(f)
            except Exception:
                cache = {}
    else:
        cache = {}
    chunks = cache.get('chunks', [])
    # Build a dict for fast lookup
    chunk_dict = {chunk['timestamp']: chunk for chunk in chunks if 'timestamp' in chunk}

    # 1. Get the latest available chunk timestamp from API (no timestamp param)
    latest_chunk_data = None
    try:
        response = requests.get(API_URL, headers=HEADERS)
        response.raise_for_status()
        latest_chunk_data = response.json()
        latest_ts_epoch = latest_chunk_data.get('timestamp')
        if latest_ts_epoch:
            latest_dt = datetime.datetime.fromtimestamp(latest_ts_epoch, datetime.timezone.utc)
            end_dt = get_5m_chunk_timestamp(latest_dt)
        else:
            # Fallback: use now rounded down to last 5m
            now = datetime.datetime.now(datetime.timezone.utc)
            end_dt = get_5m_chunk_timestamp(now)
    except Exception as e:
        print(f'Failed to fetch latest chunk: {e}')
        now = datetime.datetime.now(datetime.timezone.utc)
        end_dt = get_5m_chunk_timestamp(now)

    # 2. Compute the 36 expected chunk timestamps for the last 3 hours
    chunk_times = [end_dt - datetime.timedelta(minutes=5*i) for i in range(35, -1, -1)]
    chunk_strs = [dt.strftime('%Y-%m-%dT%H:%M:%S') for dt in chunk_times]

    # 3. Remove stale chunks outside this window.
    chunk_dict = {k: v for k, v in chunk_dict.items() if k in chunk_strs}

    # Seed the window with the newest payload from the no-timestamp endpoint.
    latest_window_ts = end_dt.strftime('%Y-%m-%dT%H:%M:%S')
    if latest_chunk_data:
        chunk_dict[latest_window_ts] = {'timestamp': latest_window_ts, 'data': latest_chunk_data}

    # 4. Fetch and add only missing chunks (parallel).
    missing = [(dt, ts_str) for dt, ts_str in zip(chunk_times, chunk_strs) if ts_str not in chunk_dict]
    if missing:
        def _fetch(item):
            dt, ts_str = item
            _ts_str_fetched, chunk_data = fetch_5m_chunk(dt)
            return ts_str, chunk_data

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_fetch, item): item[1] for item in missing}
            for future in as_completed(futures):
                ts_str = futures[future]
                try:
                    ts_str_result, chunk_data = future.result()
                    chunk_dict[ts_str] = {'timestamp': ts_str, 'data': chunk_data}
                    print(f'Fetched and added chunk: {ts_str}')
                except Exception as e:
                    print(f'Failed to fetch chunk {ts_str}: {e}')

    # 5. Save back only the active 36-chunk window, sorted by timestamp.
    final_chunks = [chunk_dict[ts_str] for ts_str in chunk_strs if ts_str in chunk_dict]

    with open(FIVE_M_FILE, 'w', encoding='utf-8') as f:
        json.dump({'chunks': final_chunks}, f, ensure_ascii=False, indent=2)
    print(f'Updated 5m cache with {len(final_chunks)} chunks (incremental window update)')

if __name__ == '__main__':
    fetch_and_save_5m_data()
