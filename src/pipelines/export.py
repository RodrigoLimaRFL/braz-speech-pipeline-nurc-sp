import locale
from logging import DEBUG
from pathlib import Path
from tqdm import tqdm
import locale
from pandas import DataFrame
import pandas as pd
from typing import List, Optional
import os

from src.services.exporter import Exporter

from src.clients.database import Database
from src.models.file import AudioFormat, File
from src.clients.google_drive import GoogleDriveClient


from src.utils import logger as lg

locale.getpreferredencoding = lambda: "UTF-8"


logger = lg.get_logger(__name__)
logger.setLevel(level=DEBUG)

def check_file_exists(audio_folder_path, file_name):
    return os.path.exists(os.path.join(audio_folder_path, file_name))

def export_corpus_dataset(
    corpus_id: int,
    output_folder: Path,
    db: Database,
    debug: bool = False,
    export_audio_to_formats: List[AudioFormat] = [AudioFormat.WAV, AudioFormat.MP3],
    sample_rate: int = 48000,
    google_drive_folder_ids: Optional[List[str]] = None,
    filter_format: Optional[AudioFormat] = None,
    export_original_audios: bool = False,
    export_to_csv: bool = False,
    export_concanated_text: bool = False,
    export_speakers_text: bool = False,
    export_json_metadata: bool = False,
    export_text_grid: bool = False,
):
    '''
    temporary measure while there is no option to select specific audios
    '''

    # audios = db.get_audios_by_corpus_id(corpus_id, filter_finished=True)
    # audios = db.get_audios_nurcsp_dev()
    audios = db.get_audios_nurcsp_test()
    # audios = db.get_audios_nurcsp_train()

    if not isinstance(audios, DataFrame) or audios.empty:
        logger.info(f"No audios found for corpus {corpus_id}.")
        return

    if debug:
        audios = audios.sample(10)

    #segments = db.get_segments_by_audios_id_list(audios.id.tolist())
    segments = db.get_segments_by_audio_id_with_extra_info(audios.id.tolist())

    if not isinstance(segments, DataFrame) or segments.empty:
        logger.info(f"No segments found for corpus {corpus_id}.")
        return

    exporter = Exporter(output_folder)
    files_dict_by_name = {}
    if export_original_audios:
        assert (
            google_drive_folder_ids is not None
        ), "You must provide at least one folder ID from Google Drive for exporting original audios."
        assert (
            filter_format is not None
        ), "You must provide a format for searching the audio files in Google Drive (wav, mp3 or mp4)."

        storage_client = GoogleDriveClient()
        files: List[File] = storage_client.get_files_from_folders(
            folder_ids=google_drive_folder_ids, filter_format=filter_format
        )

        files_dict_by_name: dict[str, File] = {
            File.clean_name(file.name): file for file in files
        }

    if export_to_csv:
        logger.info(f"Exporting audios and segments for corpus {corpus_id} to csv.")
        #exporter.export_to_csv(corpus_id, audios, segments)
        exporter.export_for_asr_csv(corpus_id, segments, "test")

    if export_concanated_text or export_speakers_text or export_text_grid or export_json_metadata or export_original_audios:
        prepared_audios = audios.rename(
            columns={
                "id": "audio_id",
                "name": "audio_name",
                "duration": "audio_duration",
            }
        )
        # Joining the DataFrames
        merged_df = pd.merge(segments, prepared_audios, on="audio_id")

        # Group by audio_id
        grouped = merged_df.groupby("audio_id")

        for audio_id, group in tqdm(grouped):
            # Get the audio_name for this audio_id
            audio_name = group["audio_name"].iloc[0]
            logger.info(
                f" # Working on the export of audio {audio_name}."
            )

            # Sort the group by segment_num
            sorted_group = group.sort_values("segment_num")

            if export_json_metadata and not check_file_exists(output_folder / audio_name, f"{audio_name}_metadata.json"):
                
                logger.info(f"Exporting metadata to json.")
                audio = audios[audios.name == audio_name].iloc[0]
                exporter.export_audio_metadata(audio)
                
            if export_concanated_text and not check_file_exists(output_folder / audio_name, f"{audio_name}_concatenated_text.txt"):
                logger.info(
                    f"Exporting concatenated text file."
                )
                exporter.export_concatenated_text_file(audio_name, sorted_group)

            if export_speakers_text and not check_file_exists(output_folder / audio_name, f"{audio_name}_by_speaker.txt"):
                logger.info(f"Exporting speakers text file.")
                exporter.export_speakers_text_file(audio_name, sorted_group)

            if export_text_grid and not check_file_exists(output_folder / audio_name, f"{audio_name}.textgrid"):
                logger.info(f"Exporting text grid file.")
                exporter.export_textgrid_file(audio_name, sorted_group)

            if export_original_audios and (not check_file_exists(output_folder / audio_name, f"{audio_name}.wav") or not check_file_exists(output_folder / audio_name, f"{audio_name}.mp3")):
                assert (
                    google_drive_folder_ids is not None
                ), "You must provide at least one folder ID from Google Drive for exporting original audios."
                assert (
                    filter_format is not None
                ), "You must provide a format for searching the audio files in Google Drive (wav, mp3 or mp4)."

                logger.info(
                    f"Exporting original audio."
                )
                exporter.export_original_audios(
                    audio_name,
                    files_dict_by_name,
                    sample_rate,
                    export_audio_to_formats,
                )
