from pathlib import Path
import pandas as pd
import textgrid
from typing import List
import soundfile as sf
import json
import re

from src.models.file import AudioFormat, File
from src.clients.google_drive import GoogleDriveClient
from src.services.audio_loader_service import AudioLoaderService
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Exporter:
    def __init__(self, output_folder: Path):
        output_folder.mkdir(parents=True, exist_ok=True)
        self.output_folder = output_folder

    def export_to_csv(
        self, corpus_id: int, audios: pd.DataFrame, segments: pd.DataFrame
    ):
        audios.to_csv(
            self.output_folder / f"corpus_{corpus_id}_audios.csv", index=False
        )
        segments.to_csv(
            self.output_folder / f"corpus_{corpus_id}_segments.csv", index=False
        )
        segments.to_parquet(
            self.output_folder / f"corpus_{corpus_id}_segments.parquet", index=False
        )
        pass

        
    def analyse_json_metadata(self, metadata: str):
        '''
        Analyse the JSON metadata (saved as a str) of an audio

        Args:
            metadata (str): The metadata

        Returns:
            dict: The analysed metadata
        '''
        analysed_metadata = {}
        # Parse JSON string into a dictionary
        metadata_dict = json.loads(metadata)

        # Get the keys from the dictionary
        keys = metadata_dict.keys()

        # Transform keys into a string
        keys_string = ', '.join(keys)

        # Check if the keys are in the metadata
        if "sexo" in keys:
            analysed_metadata["sex"] = metadata_dict["sexo"]
        else:
            analysed_metadata["sex"] = "unknown"

        if "faixa_etaria" in keys:
            analysed_metadata["age_range"] = metadata_dict["faixa_etaria"]
        else:
            analysed_metadata["age_range"] = "unknown"
        
        return analysed_metadata

    def export_for_asr_csv(self, corpus_id: int, segments: pd.DataFrame, division: str = "test"):
        '''
        Export the segments to a CSV file for ASR

        Args:
            corpus_id (int): The corpus ID
            segments (pd.DataFrame): The segments DataFrame
            division (str, optional): The division name. Defaults to "test".
        '''
        # Substrings to remove from the text
        substrings = ["(risos)", "(gargalhadas)", "(", ")"]
        # Add new columns to the segments DataFrame
        segments['quality'] = ""
        segments["speech_genre"] = ""
        segments["sex"] = ""
        segments["age_range"] = ""
        modified_rows = []
        for index, row in segments.iterrows():
            row['quality'] = 'high'
            # Check if the value in the 'text' column is '###'
            if row['text'] == '###':
                # If so, drop the row
                continue
            
            # strip the text so the whitespace does not interfere
            trimmed_text = row['text'].strip()

            # Remove the substrings from the text
            for substring in substrings:
                if substring in trimmed_text:
                    trimmed_text = trimmed_text.replace(substring, "")
                    row['quality'] = 'low'

            trimmed_text = trimmed_text.strip()

            # Remove all truncated words
            if trimmed_text.endswith('>') or trimmed_text.endswith('<'):
                trimmed_text = ' '.join(trimmed_text.split()[:-1])
                row['quality'] = 'low'

            if trimmed_text.startswith('<') or trimmed_text.startswith('>'):
                trimmed_text = ' '.join(trimmed_text.split()[1:])
                row['quality'] = 'low'

            trimmed_text = trimmed_text.replace(">", "")
            trimmed_text = trimmed_text.replace("<", "")

            trimmed_text = trimmed_text.strip()

            row['text'] = trimmed_text


            # Extract the audio type from the audio name
            genre = 'not_defined'
            pattern = r'^SP_(D2|DID|EF)_' #regex pattern to extract the audio type
            match = re.search(pattern, row['audio_name'])

            if match:
                if match.group(1) == 'D2':
                    genre = 'dialogue'
                elif match.group(1) == 'DID':
                    genre = 'interview'
                elif match.group(1) == 'EF':
                    genre = 'lecture and talks'

            row['speech_genre'] = genre


            json_metadata = Exporter.analyse_json_metadata(self, row['json_metadata'])

            row['sex'] = json_metadata['sex']
            row['age_range'] = json_metadata['age_range']


            modified_rows.append(row)

        modified_segments = pd.DataFrame(modified_rows)
        df = modified_segments[["audio_name", "file_path", "text", "start_time", "end_time", "sex", "age_range", "speech_genre", "num_speakers", "speaker_id", "quality"]]
        # duration in seconds
        df["duration"] = (df["end_time"] - df["start_time"]).round(3)
        # language variety
        df["variety"] = 'pt-br'
        # accent
        df["accent"] = 'sp-city'
        # speech style
        df["speech_style"] = "spontaneous speech"

        df = df[["audio_name", "file_path", "text", "start_time", "end_time", "duration", "quality", "speech_genre", "speech_style", "variety", "accent", "sex", "age_range", "num_speakers", "speaker_id"]]

        df.to_csv(
            self.output_folder / f"corpus_{corpus_id}_{division}.csv", index=False
        )
        pass

    def export_concatenated_text_file(self, audio_name: str, group):
        # sorted_group = group.sort_values('segment_num')

        # Concatenate the text
        concatenated_text = " ".join(group["text"]).replace("\n", "")

        output_file_path = self.output_folder / audio_name
        output_file_path.mkdir(parents=True, exist_ok=True)

        # Write to file
        with open(
            output_file_path / f"{audio_name}_concatenated_text.txt", "w"
        ) as file:
            file.write(concatenated_text)

    def export_speakers_text_file(self, audio_name: str, group):
        # Sort the group by segment_num, if it's not already sorted
        group = group.sort_values("segment_num")

        # Initialize variables
        current_speaker = None
        formatted_text = ""
        current_speaker_text = []

        for _, row in group.iterrows():
            speaker_id = row["speaker_id"]
            text = row["text"]

            # If speaker_id is null, use the current speaker, or default to 0 if it's the first segment
            speaker_id = (
                speaker_id
                if pd.notnull(speaker_id)
                else (current_speaker if current_speaker else 0)
            )

            # If the speaker has changed or it's the first segment, append the current speaker's text to formatted_text
            if current_speaker is not None and current_speaker != speaker_id:
                formatted_text += f'SPEAKER {int(current_speaker)}: {" ".join(current_speaker_text)}\n\n'
                current_speaker_text = (
                    []
                )  # Reset current_speaker_text for the new speaker

            current_speaker = speaker_id  # Update the current speaker
            current_speaker_text.append(
                text
            )  # Append the text to the current speaker's text

        # Append the last speaker's text
        formatted_text += (
            f'SPEAKER {int(current_speaker or 0)}: {" ".join(current_speaker_text)}\n'
        )

        output_file_path = self.output_folder / audio_name
        output_file_path.mkdir(parents=True, exist_ok=True)

        # Write to file
        with open(output_file_path / f"{audio_name}_by_speaker.txt", "w") as file:
            file.write(formatted_text)

    def export_textgrid_file(self, audio_name: str, group):
        # Create a new TextGrid object
        tg = textgrid.TextGrid(
            name=audio_name, minTime=0, maxTime=group["audio_duration"].iloc[0]
        )

        # Group by speaker_id
        speaker_groups = group.groupby("speaker_id", sort=False)

        for speaker_id, speaker_group in speaker_groups:
            # Corrected line: get xmax from the max end_time of the speaker_group
            tier = textgrid.IntervalTier(
                name=f"SPEAKER {int(speaker_id)}"
                if pd.notnull(speaker_id)
                else "SPEAKER 1",
                minTime=0,
                maxTime=speaker_group["end_time"].max(),
            )

            for _, row in speaker_group.iterrows():
                interval = textgrid.Interval(
                    row["start_time"], row["end_time"], row["text"]
                )
                # Add an interval for each segment
                tier.addInterval(interval)

            # Add the tier to the TextGrid
            tg.append(tier)

        output_file_path = self.output_folder / audio_name
        output_file_path.mkdir(parents=True, exist_ok=True)

        # Write the TextGrid to a file
        with open(output_file_path / f"{audio_name}.textgrid", "w") as file:
            tg.write(file)

    def export_audio_metadata(self, audio: pd.Series):
        audio_name = audio["name"]

        metadata = audio["json_metadata"]
        if metadata is None:
            metadata = {}
        else:
            metadata = json.loads(metadata)

        output_file_path = self.output_folder / audio_name
        output_file_path.mkdir(parents=True, exist_ok=True)

        with open(
            output_file_path / f"{audio_name}_metadata.json", "w", encoding="utf-8"
        ) as file:
            json.dump(metadata, file, ensure_ascii=False, indent=4)

    def export_original_audios(
        self,
        audio_name: str,
        all_files: dict[str, File],
        sample_rate: int,
        target_formats: List[AudioFormat],
    ):
        storage_client = GoogleDriveClient()


        # Get the audio file from Google Drive
        audio_file = all_files.get(File.clean_name(audio_name), None)

        if audio_file is None:
            logger.warning(
                f"Audio {audio_name} not found in GoogleDrive provided folders. Skipping."
            )
            return

        logger.debug(
            f"Audio {audio_name} found in GoogleDrive provided folders. Processing."
        )
        # Load the audio file
        audio = AudioLoaderService(storage_client).load_audio(
            audio_file, sample_rate, mono_channel=True, normalize=False
        )

        output_file_path = self.output_folder / audio_name
        output_file_path.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Audio {audio_name} loaded. Converting to target formats.")
        # Convert the audio file to the target formats
        for target_format in target_formats:
            sf.write(
                output_file_path / f"{audio_name}.{target_format.value}",
                audio.bytes,
                sample_rate,
            )
