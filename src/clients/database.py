import pandas as pd
import pymysql
import logging
import sshtunnel
from typing import List
from sshtunnel import SSHTunnelForwarder
import os

from src.config import CONFIG
from src.models.segment import SegmentCreateInDB


class Database:
    def __enter__(self, with_ssh: bool = CONFIG.mysql.use_ssh):
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

    def _run_query(self, sql_query, params=None):
        """Runs a given SQL query via the global database connection.

        :param sql: MySQL query
        :return: Pandas DataFrame containing results for SELECT queries,
                last inserted ID for INSERT queries, None for other queries
        """
        if sql_query.strip().lower().startswith("select"):
            return pd.read_sql_query(sql_query, self.sql_connection, params=params)  # type: ignore
        else:
            with self.sql_connection.cursor() as cursor:
                cursor.execute(sql_query, params)
                self.sql_connection.commit()
                if sql_query.strip().lower().startswith("insert"):
                    return cursor.lastrowid

    def add_audio(self, audio_name: str, corpus_id: int, duration: float) -> int:
        query = """
    INSERT INTO Audio
        (
            name, corpus_id, duration
        )
    VALUES
        (
            %s, %s, %s
        )
    """
        params = (audio_name, corpus_id, duration)
        audio_id = self._run_query(query, params)
        return audio_id  # type: ignore

    def add_audio_segment(self, segment: SegmentCreateInDB):
        query = """
            INSERT INTO Dataset 
            (
                file_path, file_with_user, data_gold, task, 
                text_asr, audio_id, segment_num,
                audio_lenght, duration, start_time, end_time, speaker_id
            )
            VALUES 
            (
                %s, 0, 0, 1, 
                %s, %s, %s,
                %s, %s, %s, %s, %s
            )
        """
        params = (
            segment.segment_path,
            segment.text_asr,
            segment.audio_id,
            segment.segment_num,
            segment.frames,
            segment.int_duration,
            segment.start_time,
            segment.end_time,
            segment.speaker,
        )
        return self._run_query(query, params)

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

    def get_audios_by_corpus_id(self, corpus_id, filter_finished=False):
        query = f"""
        SELECT *
        FROM Audio
        WHERE corpus_id = {corpus_id}
        AND (error_flag is null OR error_flag = 0 OR error_flag = false)
        { f"AND finished >= 1" if filter_finished else ""}
        """
        return self._run_query(query)
    
    def get_audios_nurcsp_test(self):
        '''
        Get the chosen test audios for the NURC-SP corpus

        Returns:
            pd.DataFrame: The DataFrame with the audios
        '''

        query = f"""
        SELECT *
        FROM Audio
        WHERE corpus_id = 2
        AND (error_flag is null OR error_flag = 0 OR error_flag = false)
        AND name IN ('SP_EF_363',
        'SP_EF_154',
        'SP_EF_106',
        'SP_EF_057',
        'SP_EF_187',
        'SP_D2_169',
        'SP_D2_334',
        'SP_D2_342',
        'SP_DID_271',
        'SP_DID_004',
        'SP_DID_019',
        'SP_DID_027',
        'SP_DID_200',
        'SP_DID_279'
        )
        """

        return self._run_query(query)
    
    def get_audios_nurcsp_dev(self):
        '''
        Get the chosen dev audios for the NURC-SP corpus

        Returns:
            pd.DataFrame: The DataFrame with the audios
        '''

        query = f"""
        SELECT *
        FROM Audio
        WHERE corpus_id = 2
        AND (error_flag is null OR error_flag = 0 OR error_flag = false)
        AND name IN ('SP_EF_366',
        'SP_EF_364',
        'SP_EF_375',
        'SP_D2_173',
        'SP_DID_119',
        'SP_DID_202'
        )
        """

        return self._run_query(query)
    
    def get_audios_nurcsp_train(self):
        '''
        Get the chosen train audios for the NURC-SP corpus

        Returns:
            pd.DataFrame: The DataFrame with the audios
        '''
        query = f"""
        SELECT *
        FROM Audio
        WHERE corpus_id = 2
        AND (error_flag is null OR error_flag = 0 OR error_flag = false)
        AND name NOT IN ('SP_EF_363',
        'SP_EF_154',
        'SP_EF_106',
        'SP_EF_057',
        'SP_EF_187',
        'SP_D2_169',
        'SP_D2_334',
        'SP_D2_342',
        'SP_DID_271',
        'SP_DID_004',
        'SP_DID_019',
        'SP_DID_027',
        'SP_DID_200',
        'SP_DID_279',
        'SP_EF_366',
        'SP_EF_364',
        'SP_EF_375',
        'SP_D2_173',
        'SP_DID_119',
        'SP_DID_202'
        )
        """

        return self._run_query(query)

    def get_segments_by_audio_id(self, audio_id):
        query = f"""
        SELECT *
        FROM Dataset
        WHERE audio_id = {audio_id}
        """
        return self._run_query(query)
    
    def get_segments_by_audios_id_list(self, audios_ids: List[int]):
        query = f"""
        SELECT *
        FROM Dataset
        WHERE audio_id IN ({','.join([str(id) for id in audios_ids])})
        """
        return self._run_query(query)
    
    def get_segments_by_audio_id_with_extra_info(self, audios_ids: List[int]):
        query = f"""
        SELECT 
        d.*, a.name as audio_name, a.json_metadata as json_metadata, a.num_speakers as num_speakers
        FROM Dataset d
        JOIN Audio a
        ON d.audio_id = a.id
        WHERE d.audio_id IN ({','.join([str(id) for id in audios_ids])})
        """
        return self._run_query(query)