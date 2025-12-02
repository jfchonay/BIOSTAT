from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from access_drive import get_parent_id, get_list_csv
import io
import pandas as pd


gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)
# root folder and nested folders
SHARED_ROOT_NAME = "2_Transformed and organized ALL_RAW DATA_23_missing"      # name of folder in "Shared with me"
SUBPATH = "sub-001"             # nested path inside that shared folder
# 1) resolve nested folder ID
folder_id = get_parent_id(drive, SHARED_ROOT_NAME, SUBPATH, False)
# 2) get list of CSV in that folder
list_csv = get_list_csv(drive, folder_id)
# 3) read CSV into DataFrame (in memory, no local file)
gfile = drive.CreateFile({'id': list_csv[0]['id']})
csv_str = gfile.GetContentString()             # str in RAM
df = pd.read_csv(io.StringIO(csv_str))         # DataFrame in RAM

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

