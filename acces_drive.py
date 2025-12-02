from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import io
import pandas as pd


def get_shared_folder_id_by_path(drive, shared_root_name: str, subpath: str) -> str:
    """
    Resolve a path like under a shared root folder
    (from the 'Shared with me' section) to a folder ID.
    """
    # 1) find the shared root folder
    shared_folders = drive.ListFile({
        'q': (
            "sharedWithMe = true and "
            "mimeType = 'application/vnd.google-apps.folder' and "
            f"title = '{shared_root_name}' and "
            "trashed = false"
        )
    }).GetList()

    if not shared_folders:
        raise RuntimeError(f"No shared folder named {shared_root_name!r}")

    parent_id = shared_folders[0]['id']

    # 2) walk down the nested subfolders
    parts = [p for p in subpath.split('/') if p]
    for name in parts:
        q = (
            "mimeType = 'application/vnd.google-apps.folder' and "
            f"title = '{name}' and "
            f"'{parent_id}' in parents and "
            "trashed = false"
        )
        folders = drive.ListFile({'q': q}).GetList()
        if not folders:
            raise RuntimeError(f"Folder {name!r} not found under parent {parent_id}")
        parent_id = folders[0]['id']

    return parent_id


def get_first_csv_in_folder(drive, folder_id: str):
    """
    Returns the first CSV file metadata object (or None).
    """
    file_list = drive.ListFile({
        'q': (
            f"'{folder_id}' in parents and "
            "mimeType = 'text/csv' and "
            "trashed = false"
        )
    }).GetList()

    if not file_list:
        return None

    # you can sort here if you need deterministic order (e.g. by title)
    # file_list.sort(key=lambda f: f['title'])
    return file_list[0]

# --- auth (you already have this) ---
gauth = GoogleAuth()
gauth.Authorize()
drive = GoogleDrive(gauth)
# --- parameters you care about ---
SHARED_ROOT_NAME = "PM Magda Jose Sonja"      # name of folder in "Shared with me"
SUBPATH = "job during holiday crosschecking/original csv"             # nested path inside that shared folder

# 1) resolve nested folder ID
folder_id = get_shared_folder_id_by_path(drive, SHARED_ROOT_NAME, SUBPATH)
print("Target folder ID:", folder_id)

# 2) get first CSV in that folder
first_csv = get_first_csv_in_folder(drive, folder_id)
if first_csv is None:
    raise RuntimeError("No CSV files found in target folder")

print("Using CSV:", first_csv["title"], "ID:", first_csv["id"])

# 3) read CSV into DataFrame (in memory, no local file)
gfile = drive.CreateFile({'id': first_csv['id']})
csv_str = gfile.GetContentString()             # str in RAM
df = pd.read_csv(io.StringIO(csv_str))         # DataFrame in RAM

# 4) append a row that says 'test'
#    Generic strategy: put 'test' in the first column, NaN in others
new_row = {col: ( 'test' if i == 0 else pd.NA ) for i, col in enumerate(df.columns)}
df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

# 5) serialize modified DataFrame to CSV string (in memory)
buf = io.StringIO()
df.to_csv(buf, index=False)
modified_csv_str = buf.getvalue()

# 6) write back to Drive as a *new* file in the same folder
new_title = f"modified_{first_csv['title']}"

g_new = drive.CreateFile({
    "title": new_title,
    "mimeType": "text/csv",
    "parents": [{"id": folder_id}],
})
g_new.SetContentString(modified_csv_str)
g_new.Upload()

print("Created modified file:", new_title, "ID:", g_new["id"])