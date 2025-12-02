def get_parent_id(drive, root_name: str, subpath: str = None, is_shared: bool = False) -> str:
    """
    Resolve a nested folder path to a Drive folder ID.

    Parameters
    ----------
    drive : GoogleDrive
        Authenticated PyDrive2 client
    root_name : str
        Name of the top-level folder
    subpath : str, optional
        Nested path under the root
        If None or empty, return the root folder ID directly.
    is_shared : bool
        True  -> search in 'Shared with me'
        False -> search in 'My Drive'

    Returns
    -------
    str : Google Drive folder ID
    """

    # --- 1) locate the root folder ---
    if is_shared:
        root_query = (
            "sharedWithMe = true and "
            "mimeType = 'application/vnd.google-apps.folder' and "
            f"title = '{root_name}' and "
            "trashed = false"
        )
    else:
        root_query = (
            "mimeType = 'application/vnd.google-apps.folder' and "
            f"title = '{root_name}' and "
            "'root' in parents and "
            "trashed = false"
        )

    roots = drive.ListFile({'q': root_query}).GetList()
    if not roots:
        location = "Shared with me" if is_shared else "My Drive"
        raise RuntimeError(f"Folder '{root_name}' not found in {location}.")

    parent_id = roots[0]['id']

    # --- 2) if no subpath, return the root folder directly ---
    if not subpath:
        return parent_id

    # --- 3) walk down nested subfolders ---
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
            raise RuntimeError(f"Folder '{name}' not found under parent {parent_id}")
        parent_id = folders[0]['id']

    return parent_id


def get_list_csv(drive, folder_id: str):
    """
        Returns the list of CSV file metadata object (or None).
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
    return file_list