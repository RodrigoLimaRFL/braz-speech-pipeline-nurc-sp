import paramiko
from scp import SCPClient
import os
import pipes

from src.config import CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FileTransfer:
    def __enter__(self):
        self.ssh = self._create_ssh_client(
            CONFIG.sshtunnel.host,
            CONFIG.sshtunnel.port,
            CONFIG.sshtunnel.username,
            CONFIG.sshtunnel.password,
        )
        transport = self.ssh.get_transport()
        if transport is None or not transport.is_active():
            raise Exception("SSH connection is not active.")

        self.scp = SCPClient(transport)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.scp.close()
        self.ssh.close()

    def _create_ssh_client(self, server, port, user, password):
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(server, port, user, password)
        return client

    def mkdir(self, folder_path: str, recursive: bool = False):
        command = f"printf '%q ' mkdir -p {pipes.quote(folder_path)}"  # Ignore error if the folder already exists
        stdin, stdout, stderr = self.ssh.exec_command(command)

        # exec_command() returns file-like objects representing the input (stdin),
        # output (stdout) and error (stderr) channels from the SSH session.
        # You may need to call read() or readlines() on these objects to retrieve the actual command output.
        out = stdout.readlines()
        err = stderr.readlines()

        injected_command = "".join(out)
        # reinterpret printf output as a command

        # command = f"mkdir -p -v {folder_path}"  # Ignore error if the folder already exists
        self.ssh.exec_command(injected_command)

    def put(self, source: str, target: str, target_is_folder: bool = False, **kwargs):
        if target_is_folder:
            self.mkdir(target)
            filename = os.path.basename(source)
            target = os.path.join(target, filename)
        else:
            target_directory = os.path.dirname(target)
            self.mkdir(target_directory)

        try:
            self.scp.put(source, target.encode(), **kwargs)
        except Exception as e:
            logger.error(f"Error transfering file.")
            logger.debug(e)
        else:
            logger.debug(f"File transfered successfully.")

    def read_all_files(self, path):
        command = f"ls {path}"
        stdin, stdout, stderr = self.ssh.exec_command(command)
        print(stderr)
        return stdout.readlines()

    def read_all_files_in_folder_and_subfolders(self, path):
        command = f"find {path} -type f"
        stdin, stdout, stderr = self.ssh.exec_command(command)
        print(stderr)
        return stdout.readlines()
