for folder in utterances context
do
    videos_folder_path=data/videos/"${folder}"_final
    audios_folder_path=data/audios/"${folder}"_final
    ext=mp4

    mkdir -p "${audios_folder_path}"

    for video_file_path in "${videos_folder_path}"/*."${ext}"; do
        slash_and_video_file_name="${video_file_path:${#videos_folder_path}}"
        slash_and_video_file_name_without_extension="${slash_and_video_file_name%.${ext}}"
        audio_file_path="${audios_folder_path}${slash_and_video_file_name_without_extension}.wav"
        ffmpeg -i "${video_file_path}" -ac 1 "${audio_file_path}"
    done
done