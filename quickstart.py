from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

gauth = GoogleAuth(settings_file="settings.yaml")
# Try to load existing credentials
gauth.LoadCredentialsFile("credentials.json")
if gauth.credentials is None:
    # No credentials yet -> do interactive login once
    gauth.LocalWebserverAuth()
elif gauth.access_token_expired:
    # Existing but expired -> refresh
    gauth.Refresh()
else:
    # Credentials are fine -> just use them
    gauth.Authorize()
# Always save back (it will update tokens etc.)
gauth.SaveCredentialsFile("credentials.json")
drive = GoogleDrive(gauth)