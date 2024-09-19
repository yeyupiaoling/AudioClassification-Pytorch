import tarfile, io, os

class TarDataset:
    def __init__(self, tar_path):
        self.tar_path = tar_path
        if not os.path.exists(self.tar_path):
            with tarfile.open(self.tar_path, 'w') as tar:
                pass  # Create an empty tar file
            self.tar = tarfile.open(self.tar_path, 'a')
        else:
            self.tar = tarfile.open(self.tar_path, self._get_mode())
    
    def _get_mode(self):
        return 'r:gz' if self.tar_path.endswith('.tar.gz') else 'r'

    def list_files(self):
        return [member.name for member in self.tar.getmembers()]

    def read_file(self, file_path):
        return self.tar.extractfile(file_path).read()

    def write_file(self, file_path, data):
        tarinfo = tarfile.TarInfo(name=file_path)
        if isinstance(data, (bytes, bytearray)):
            tarinfo.size = len(data)
            self.tar.addfile(tarinfo, io.BytesIO(data))
        elif hasattr(data, 'read'):
            data.seek(0, os.SEEK_END)
            tarinfo.size = data.tell()
            data.seek(0)
            self.tar.addfile(tarinfo, data)
        else:
            raise TypeError("data must be bytes-like or a file-like object")

    def close(self):
        self.tar.close()