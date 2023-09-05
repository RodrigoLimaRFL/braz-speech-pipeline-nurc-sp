import pandas as pd
import pymysql
import logging
import sshtunnel
from sshtunnel import SSHTunnelForwarder
import os

from src.config import CONFIG
from src.models.segment import Segment


class Database:
    def __enter__(self, with_ssh: bool = True):
        self.ssh = self._open_ssh_tunnel() if with_ssh else None
        self.sql_connection = self._mysql_connect()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sql_connection.close()
        if self.ssh is not None:
            self.ssh.close()

    def _open_ssh_tunnel(self, verbose=False):
        """Open an SSH tunnel and connect using a username and password.

        :param verbose: Set to True to show logging
        :return tunnel: Global SSH tunnel connection
        """

        if verbose:
            sshtunnel.DEFAULT_LOGLEVEL = logging.DEBUG

        tunnel = SSHTunnelForwarder(
            (CONFIG.sshtunnel.host, CONFIG.sshtunnel.port),
            ssh_username=CONFIG.sshtunnel.username,
            ssh_password=CONFIG.sshtunnel.password,
            remote_bind_address=("127.0.0.1", 3306),
        )

        tunnel.start()

        return tunnel

    def _mysql_connect(self):
        """Connect to a MySQL server using the SSH tunnel connection

        :return connection: Global MySQL database connection
        """
        connection = pymysql.connect(
            host=CONFIG.mysql.host,
            user=CONFIG.mysql.username,
            passwd=CONFIG.mysql.password,
            db=CONFIG.mysql.database,
            port=self.ssh.local_bind_port
            if self.ssh is not None
            else CONFIG.mysql.port,
        )

        return connection

    def _run_query(self, sql):
        """Runs a given SQL query via the global database connection.

        :param sql: MySQL query
        :return: Pandas DataFrame containing results for SELECT queries,
                last inserted ID for INSERT queries, None for other queries
        """
        if sql.strip().lower().startswith("select"):
            return pd.read_sql_query(sql, self.sql_connection)  # type: ignore
        else:
            with self.sql_connection.cursor() as cursor:
                cursor.execute(sql)
                self.sql_connection.commit()
                if sql.strip().lower().startswith("insert"):
                    return cursor.lastrowid

    def add_audio(self, audio_name: str, corpus_id: int, duration: float) -> int:
        query = f"""
    INSERT INTO Audio
        (
            name, corpus_id, duration
        )
    VALUES
        (
            '{audio_name}', {corpus_id}, {duration}
        )
    """
        audio_id = self._run_query(query)
        return audio_id  # type: ignore

    def add_audio_segment(
        self,
        segment: Segment
    ):
        query = f"""
    INSERT INTO Dataset 
        (
            file_path, file_with_user, data_gold, task, 
            text_asr, audio_id, segment_num,
            audio_lenght, duration, start_time, end_time, speaker_id
        )
    VALUES 
        (
            '{segment.segment_path}', 0, 0, 1, 
            '{segment.text_asr}', {segment.audio_id}, {segment.segment_num},
            {segment.frames}, {segment.duration}, {segment.start_time}, {segment.end_time}, {segment.speaker_id}
        )
    """
        return self._run_query(query)

    def update_audio_duration(self, audio_id, audio_duration):
        query = f"""
        UPDATE Audio
        SET duration = {audio_duration}
        WHERE id = {audio_id}
        """
        return self._run_query(query)

    def get_audios_by_name(self, audio_name):
        query = f"""
        SELECT *
        FROM Audio
        WHERE name LIKE '{audio_name}%'
        """
        return self._run_query(query)
