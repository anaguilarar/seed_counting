import zipfile
from io import BytesIO
import requests
from urllib.parse import urlparse
import os



def downloadzip(urlpath, foldername = 'models')-> None: 
    """
    function to pull a zip file from internet

    Parameters:
    --------
    urlpath: str
        url link which contian the file
    
    foldername: str
        the folder name in which the extracted file will be located
    
    Returrn:
    --------
    None


    """
    if foldername is None:
        foldername = ""

    if urlpath.startswith('http'):
        a = urlparse(urlpath)
        
        if not os.path.exists(os.path.basename(a.path)):
            req = requests.get(urlpath)

            with zipfile.ZipFile(BytesIO(req.content)) as zipobject:
                zipobject.extractall(foldername)
        
        else:
            zipobject = zipfile.ZipFile(os.path.basename(a.path))
            if not os.path.exists(os.path.join(foldername,
                zipobject.filelist[0].filename)):
                with zipfile.ZipFile(os.path.basename(a.path)) as zipobject:
                    zipobject.extractall(foldername)


def filter_files_usingsuffix(path, suffix = 'h5'):
    """
    function to pull a zip file from internet

    Parameters:
    --------
    path: str
        path that contains the files
    
    suffix: str
        use a string to filter the files that are inside the extracted folder
    

    Returrn:
    --------

    path to the file

    """

    fileinfolder = [i for i in os.listdir(path) if i.endswith(suffix)]

    if len(fileinfolder)==1:
        wp = fileinfolder[0]
        wp = os.path.join(path, wp)
    else:
        raise ValueError("there is no files with this extension {}".format(suffix))
       
    return wp
