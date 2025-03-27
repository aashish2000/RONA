import os, json, functools, time, csv, re
root_dir = os.path.dirname(os.path.abspath(__file__))
from PIL import Image

def save_json(data, path):
    folder = os.path.dirname(path)
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
        
def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def write_log(log_path, text):
    folder = os.path.dirname(log_path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(log_path, 'a') as f:
        f.write(text)

def read_csv(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

def write_csv(csv_path, data):
    folder = os.path.dirname(csv_path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data)

def clean_json_string(json_string, type_of_json="object"):
    pattern = r'^\s*```json\s*(.*?)\s*```\s*$'
    cleaned_string = re.sub(pattern, r'\1', json_string, flags=re.DOTALL)
    other_pattern = r'^\s*```\s*(.*?)\s*```\s*$'
    cleaned_string = re.sub(other_pattern, r'\1', cleaned_string, flags=re.DOTALL)
    if type_of_json == "array":
        match_pattern = r'\[[\w\W]+?\]'
    else:
        match_pattern = r'{[\w\W]+?}'
    matches = re.findall(match_pattern, cleaned_string)
    if (not matches):
        if (type_of_json == "array"): return "[]"
        return "{}"
    return matches[0]

def get_image(name):
    extensions = ['jpg', 'webp', 'jpeg', 'png', 'JPG', 'Jpeg']
    found_image = None
    for ext in extensions:
        try:
            image_path = name+f'.{ext}'
            found_image = Image.open(image_path).convert('RGB')
            break
        except FileNotFoundError:
            continue

    if found_image is None:
        print(f"No valid image found for {name} !")
    return found_image

def retry_if_fail(func):
    @functools.wraps(func)
    def wrapper_retry(*args, **kwargs):
        retry = 0
        while retry <= 10:
            try:
                out = func(*args, **kwargs)
                break
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                retry += 1
                time.sleep(10)
                print(f"Exception occurred: {type(e).__name__}, {e.args}")
                print(f"Retry {retry} times...")

        if retry > 10:
            raise RuntimeError("Failed after 10 retries.")
        return out
    return wrapper_retry