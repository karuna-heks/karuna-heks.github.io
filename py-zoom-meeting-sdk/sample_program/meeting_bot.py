import os
import sys
import pathlib
from datetime import datetime, timedelta
import json
import requests

import wave
import zoom_meeting_sdk as zoom
import jwt
from deepgram_transcriber import DeepgramTranscriber
import cv2
import numpy as np
import gi
gi.require_version('GLib', '2.0')
from gi.repository import GLib


from typing import Final

TOKEN_URL:   Final = "https://zoom.us/oauth/token"
USER_TOKEN:  Final = "https://api.zoom.us/v2/users/{user_id}/token"

meeting_event_log = []

class ZoomAuthError(RuntimeError):
    """Любое отклонение Zoom OAuth."""

def _s2s_access_token(account_id: str, client_id: str, client_secret: str) -> str:
    """Шаг 1 — access_token из Server-to-Server OAuth (TTL ≈ 1 ч)."""
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type":  "account_credentials",
            "account_id":  account_id.strip(),
        },
        auth=(client_id.strip(), client_secret.strip()),
        timeout=10,
    )
    if resp.status_code != 200:
        raise ZoomAuthError(
            f"S2S OAuth failed {resp.status_code}: {resp.text.strip()}"
        )
    return resp.json()["access_token"]


def save_yuv420_frame_as_png(frame_bytes, width, height, output_path):
    try:
        # Convert bytes to numpy array
        yuv_data = np.frombuffer(frame_bytes, dtype=np.uint8)

        # Reshape into I420 format with U/V planes
        yuv_frame = yuv_data.reshape((height * 3//2, width))

        # Convert from YUV420 to BGR
        bgr_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)

        # Save as PNG
        cv2.imwrite(output_path, bgr_frame)
    except Exception as e:
        print(f"Error saving frame to {output_path}: {e}")


def generate_jwt(client_id, client_secret):
    iat = datetime.utcnow()
    exp = iat + timedelta(hours=24)

    payload = {
        "iat": iat,
        "exp": exp,
        "appKey": client_id,
        "tokenExp": int(exp.timestamp())
    }

    token = jwt.encode(payload, client_secret, algorithm="HS256")
    return token


def get_zak(client_id, client_secret, account_id):

    # 1. S2S OAuth access token
    resp = requests.post(
        "https://zoom.us/oauth/token",
        data = {"grant_type":"account_credentials",
                "account_id": account_id},
        auth = (client_id, client_secret))
    resp.raise_for_status()
    access_token = resp.json()["access_token"]

    # 2. ZAK for the bot-user (self)
    zak = requests.get(
        "https://api.zoom.us/v2/users/me/token?type=zak",
        headers={"Authorization": f"Bearer {access_token}"}).json()["token"]
    
    return zak



def normalized_rms_audio(pcm_data: bytes, sample_width: int = 2) -> bool:
    """
    Determine if PCM audio data contains significant audio or is essentially silence.
    
    Args:
        pcm_data: Bytes object containing PCM audio data in linear16 format
        threshold: RMS amplitude threshold below which audio is considered silent (0.0 to 1.0)
        sample_width: Number of bytes per sample (2 for linear16)
        
    Returns:
        bool: True if the audio is essentially silence, False if it contains significant audio
    """
    if len(pcm_data) == 0:
        return True
        
    # Convert bytes to 16-bit integers
    import array
    samples = array.array('h')  # signed short integer array
    samples.frombytes(pcm_data)
    
    # Calculate RMS amplitude
    sum_squares = sum(sample * sample for sample in samples)
    rms = (sum_squares / len(samples)) ** 0.5
    
    # Normalize RMS to 0.0-1.0 range
    # For 16-bit audio, max value is 32767
    normalized_rms = rms / 32767.0
    return normalized_rms


def create_red_yuv420_frame(width=640, height=360):
    # Create BGR frame (red is [0,0,255] in BGR)
    bgr_frame = np.zeros((height, width, 3), dtype=np.uint8)
    bgr_frame[:, :] = [0, 0, 255]  # Pure red in BGR
    
    # Convert BGR to YUV420 (I420)
    yuv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YUV_I420)
    
    # Return as bytes
    return yuv_frame.tobytes()


class MeetingBot:

    def __init__(self, meeting_id: str, secret: str):
        self.meeting_id = meeting_id
        self.secret = secret
        self.client_id = os.environ.get("ZOOM_CLIENT_ID")
        self.client_secret = os.environ.get("ZOOM_CLIENT_SECRET")
        self.account_id = os.environ.get("ZOOM_ACCOUNT_ID")

        self.meeting_name = f"{self.meeting_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.meeting_service = None
        self.setting_service = None
        self.auth_service = None

        self.auth_event = None
        self.recording_event = None
        self.meeting_service_event = None

        self.audio_source = None
        self.audio_helper = None

        self.audio_settings = None

        self.use_audio_recording = True
        self.use_video_recording = os.environ.get('RECORD_VIDEO') == 'true'

        self.reminder_controller = None

        self.recording_ctrl = None
        self.audio_ctrl = None
        self.audio_ctrl_event = None
        self.audio_raw_data_sender = None
        self.virtual_audio_mic_event_passthrough = None

        self.deepgram_transcriber = DeepgramTranscriber()

        self.my_participant_id = None
        self.other_participant_id = None
        self.participants_ctrl = None
        self.meeting_reminder_event = None
        self.audio_print_counter = 0

        self.video_helper = None
        self.renderer_delegate = None
        self.video_frame_counter = 0

        self.meeting_video_controller = None
        self.video_sender = None
        self.virtual_camera_video_source = None
        self.video_source_helper = None

        self.meeting_sharing_controller = None
        self.meeting_share_ctrl_event = None

        self.share_helper = None
        self.share_video_renderer_delegate = None
        self.share_audio_renderer_delegate = None
        self.share_video_sender = None
        self.share_audio_sender = None

        self.chat_ctrl = None
        self.chat_ctrl_event = None

        self.mix_wav: wave.Wave_write | None = None      # общий файл
        self.user_wavs: dict[int, wave.Wave_write] = {}  # per-user


    def cleanup(self):
        if self.meeting_service:
            zoom.DestroyMeetingService(self.meeting_service)
            print("Destroyed Meeting service")
        if self.setting_service:
            zoom.DestroySettingService(self.setting_service)
            print("Destroyed Setting service")
        if self.auth_service:
            zoom.DestroyAuthService(self.auth_service)
            print("Destroyed Auth service")

        if self.audio_helper:
            audio_helper_unsubscribe_result = self.audio_helper.unSubscribe()
            print("audio_helper.unSubscribe() returned", audio_helper_unsubscribe_result)

        if self.video_helper:
            video_helper_unsubscribe_result = self.video_helper.unSubscribe()
            print("video_helper.unSubscribe() returned", video_helper_unsubscribe_result)

        if self.mix_wav:
            self.mix_wav.close()
            self.mix_wav = None

        for wav in self.user_wavs.values():
            wav.close()
        self.user_wavs.clear()

        print("CleanUPSDK() called")
        zoom.CleanUPSDK()
        print("CleanUPSDK() finished")


    def init(self):
        print("ZOOM_APP_CLIENT_ID: ", os.environ['ZOOM_APP_CLIENT_ID'])
        print("ZOOM_APP_CLIENT_SECRET: ", os.environ['ZOOM_APP_CLIENT_SECRET'])
        print("MEETING_ID: ", os.environ['MEETING_ID'])
        print("MEETING_PWD: ", os.environ['MEETING_PWD'])

        if os.environ.get('ZOOM_APP_CLIENT_ID') is None:
            raise RuntimeError('No ZOOM_APP_CLIENT_ID found in environment. Please define this in a .env file located in the repository root')
        if os.environ.get('ZOOM_APP_CLIENT_SECRET') is None:
            raise RuntimeError('No ZOOM_APP_CLIENT_SECRET found in environment. Please define this in a .env file located in the repository root')

        init_param = zoom.InitParam()
        init_param.strWebDomain = "https://zoom.us"
        init_param.strSupportUrl = "https://zoom.us"
        init_param.enableGenerateDump = True
        init_param.emLanguageID = zoom.SDK_LANGUAGE_ID.LANGUAGE_English
        init_param.enableLogByDefault = True

        init_sdk_result = zoom.InitSDK(init_param)
        if init_sdk_result != zoom.SDKERR_SUCCESS:
            raise Exception('InitSDK failed')
        self.create_services()


    def on_user_join_callback(self, joined_user_ids, user_name):
        print("on_user_join_callback called. joined_user_ids =", joined_user_ids, "user_name =", user_name)


    def on_sharing_status_callback(self, share_info):
        print(
            f"on_sharing_status_callback called. ",
            f"userid = {share_info.userid} ",
            f"shareSourceID = {share_info.shareSourceID} ",
            f"status = {share_info.status} ",
            f"contentType = {share_info.contentType} ",
            f"isShowingInFirstView = {share_info.isShowingInFirstView} ",
            f"isShowingInSecondView = {share_info.isShowingInSecondView} ",
        )


    def on_failed_to_start_share_callback(self):
        print("on_failed_to_start_share_callback called")


    def on_share_content_notification_callback(self, share_info):
        print(
            f"on_share_content_notification_callback called. ",
            f"userid = {share_info.userid} ",
            f"shareSourceID = {share_info.shareSourceID} ",
            f"status = {share_info.status} ",
            f"contentType = {share_info.contentType} ",
            f"isShowingInFirstView = {share_info.isShowingInFirstView} ",
            f"isShowingInSecondView = {share_info.isShowingInSecondView} ",
        )

    def on_share_setting_type_changed_notification_callback(self, share_setting_type):
        print("on_share_setting_type_changed_notification_callback called. share_setting_type =", share_setting_type)


    def on_shared_video_ended_callback(self):
        print("on_shared_video_ended_callback called")


    def on_video_file_share_play_error_callback(self, error):
        print("on_video_file_share_play_error_callback called. error =", error)


    def on_optimizing_share_for_video_clip_status_changed_callback(self, share_info):
        print(
            f"on_optimizing_share_for_video_clip_status_changed_callback called. ",
            f"userid = {share_info.userid} ",
            f"shareSourceID = {share_info.shareSourceID} ",
            f"status = {share_info.status} ",
            f"contentType = {share_info.contentType} ",
            f"isShowingInFirstView = {share_info.isShowingInFirstView} ",
            f"isShowingInSecondView = {share_info.isShowingInSecondView} ",
        )


    # NOTE: content will always be None use chat_msg_info.GetContent() instead
    def on_chat_msg_notification_callback(self, chat_msg_info, content):
        print("\n=== on_chat_msg_notification called ===")
        print(f"Message ID: {chat_msg_info.GetMessageID()}")
        print(f"Sender ID: {chat_msg_info.GetSenderUserId()}")
        print(f"Sender Name: {chat_msg_info.GetSenderDisplayName()}")
        print(f"Receiver ID: {chat_msg_info.GetReceiverUserId()}")
        print(f"Receiver Name: {chat_msg_info.GetReceiverDisplayName()}")
        print(f"Content: {chat_msg_info.GetContent()}")
        print(f"Timestamp: {chat_msg_info.GetTimeStamp()}")
        print(f"Message Type: {chat_msg_info.GetChatMessageType()}")
        print(f"Is Chat To All: {chat_msg_info.IsChatToAll()}")
        print(f"Is Chat To All Panelist: {chat_msg_info.IsChatToAllPanelist()}")
        print(f"Is Chat To Waitingroom: {chat_msg_info.IsChatToWaitingroom()}")
        print(f"Is Comment: {chat_msg_info.IsComment()}")
        print(f"Is Thread: {chat_msg_info.IsThread()}")
        print(f"Thread ID: {chat_msg_info.GetThreadID()}")
        print("=====================\n")


    def on_join(self):
        self.meeting_reminder_event = zoom.MeetingReminderEventCallbacks(onReminderNotifyCallback=self.on_reminder_notify)
        self.reminder_controller = self.meeting_service.GetMeetingReminderController()
        self.reminder_controller.SetEvent(self.meeting_reminder_event)

        if self.use_audio_recording:
            self.recording_ctrl = self.meeting_service.GetMeetingRecordingController()

            def on_recording_privilege_changed(can_rec):
                print("on_recording_privilege_changed called. can_record =", can_rec)
                if can_rec:
                    GLib.timeout_add_seconds(1, self.start_raw_recording)
                else:
                    self.stop_raw_recording()

            self.recording_event = zoom.MeetingRecordingCtrlEventCallbacks(onRecordPrivilegeChangedCallback=on_recording_privilege_changed)
            self.recording_ctrl.SetEvent(self.recording_event)

            GLib.timeout_add_seconds(1, self.start_raw_recording)

        self.participants_ctrl = self.meeting_service.GetMeetingParticipantsController()
        self.participants_ctrl_event = zoom.MeetingParticipantsCtrlEventCallbacks(onUserJoinCallback=self.on_user_join_callback)
        self.participants_ctrl.SetEvent(self.participants_ctrl_event)
        self.my_participant_id = self.participants_ctrl.GetMySelfUser().GetUserID()

        participant_ids_list = self.participants_ctrl.GetParticipantsList()
        print("participant_ids_list", participant_ids_list)
        for participant_id in participant_ids_list:
            if participant_id != self.my_participant_id:
                self.other_participant_id = participant_id
                break
        print("other_participant_id", self.other_participant_id)

        self.meeting_sharing_controller = self.meeting_service.GetMeetingShareController()
        self.meeting_share_ctrl_event = zoom.MeetingShareCtrlEventCallbacks(
            onSharingStatusCallback=self.on_sharing_status_callback,
            onFailedToStartShareCallback=self.on_failed_to_start_share_callback,
            onShareContentNotificationCallback=self.on_share_content_notification_callback,
            onShareSettingTypeChangedNotificationCallback=self.on_share_setting_type_changed_notification_callback,
            onSharedVideoEndedCallback=self.on_shared_video_ended_callback,
            onVideoFileSharePlayErrorCallback=self.on_video_file_share_play_error_callback,
            onOptimizingShareForVideoClipStatusChangedCallback=self.on_optimizing_share_for_video_clip_status_changed_callback
        )
        self.meeting_sharing_controller.SetEvent(self.meeting_share_ctrl_event)
        viewable_sharing_user_list = self.meeting_sharing_controller.GetViewableSharingUserList()
        print("viewable_sharing_user_list", viewable_sharing_user_list)
        for user_id in viewable_sharing_user_list:
            sharing_info_list_for_user = self.meeting_sharing_controller.GetSharingSourceInfoList(user_id)
            print("sharing_info_list_for_user", user_id, " = ", sharing_info_list_for_user)

        self.audio_ctrl = self.meeting_service.GetMeetingAudioController()
        self.audio_ctrl_event = zoom.MeetingAudioCtrlEventCallbacks(onUserAudioStatusChangeCallback=self.on_user_audio_status_change_callback, onUserActiveAudioChangeCallback=self.on_user_active_audio_change_callback)
        self.audio_ctrl.SetEvent(self.audio_ctrl_event)
        # Raw audio input got borked in the Zoom SDK after 6.3.5.
        # This is work-around to get it to work again.
        # See here for more details: https://devforum.zoom.us/t/cant-record-audio-with-linux-meetingsdk-after-6-3-5-6495-error-code-32/130689/5
        self.audio_ctrl.JoinVoip()
        
        self.chat_ctrl = self.meeting_service.GetMeetingChatController()
        self.chat_ctrl_event = zoom.MeetingChatEventCallbacks(onChatMsgNotificationCallback=self.on_chat_msg_notification_callback)
        self.chat_ctrl.SetEvent(self.chat_ctrl_event)

        # Send a welcome message to the chat
        builder = self.chat_ctrl.GetChatMessageBuilder()
        builder.SetContent("Welcoome to the PyZoomMeetingSDK")
        builder.SetReceiver(0)
        builder.SetMessageType(zoom.SDKChatMessageType.To_All)
        msg = builder.Build()
        send_result = self.chat_ctrl.SendChatMsgTo(msg)
        print("send_result =", send_result)
        builder.Clear()


    def on_user_active_audio_change_callback(self, user_ids):
        print("on_user_active_audio_change_callback called. user_ids =", user_ids)
        meeting_event_log.append({
            "user_ids": user_ids,
            "event": "on_user_active_audio_change_callback",
            "ts": datetime.now().strftime("%Y.%m.%d %H:%M:%S.%f")
        })


    def on_user_audio_status_change_callback(self, user_audio_statuses, otherstuff):
        print("on_user_audio_status_change_callback called. user_audio_statuses =", 
              user_audio_statuses, "otherstuff =", otherstuff)
        meeting_event_log.append({
            "user_audio_statuses": str(user_audio_statuses),
            "event": "on_user_audio_status_change_callback",
            "otherstuff": str(otherstuff),
            "ts": datetime.now().strftime("%Y.%m.%d %H:%M:%S.%f")
        })


    def on_mic_initialize_callback(self, sender):
        print("on_mic_initialize_callback called")
        self.audio_raw_data_sender = sender
        meeting_event_log.append({
            "sender": str(sender),
            "event": "on_mic_initialize_callback",
            "ts": datetime.now().strftime("%Y.%m.%d %H:%M:%S.%f")
        })


    def on_mic_start_send_callback(self):
        print("on_mic_start_send_callback called (without audio send)")
        meeting_event_log.append({
            "event": "on_mic_start_send_callback",
            "ts": datetime.now().strftime("%Y.%m.%d %H:%M:%S.%f")
        })


    def on_one_way_audio_raw_data_received_callback(self, data, node_id):
        meeting_event_log.append({
            "event": "on_one_way_audio_raw_data_received_callback",
            "node_id": str(node_id),
            "ts": datetime.now().strftime("%Y.%m.%d %H:%M:%S.%f")
        })

        buf = data.GetBuffer()  # bytes
        # 3-a. общий микс
        if self.mix_wav:
            self.mix_wav.writeframes(buf)

        # 3-b. per-user файлы (опционно)
        if node_id not in self.user_wavs:
            out_dir = pathlib.Path(f"sample_program/out/audio/{self.meeting_name}")
            if not out_dir.exists():
                out_dir.mkdir()
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            wav_path = out_dir / f"user_{node_id}_{ts}.wav"
            wav = wave.open(str(wav_path), "wb")
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(32000)
            self.user_wavs[node_id] = wav
        self.user_wavs[node_id].writeframes(buf)


    # def on_share_audio_start_send_callback(self, sender):
    #     print("on_share_audio_start_send_callback called, sender =", sender)
    #     meeting_event_log.append({
    #         "event": "on_share_audio_start_send_callback",
    #         "sender": sender,
    #         "ts": datetime.now()
    #     })
    #     self.share_audio_sender = sender
        
    #     audio_path = 'sample_program/input_audio/test_audio_16778240.pcm'
        
    #     if not os.path.exists(audio_path):
    #         print(f"Audio file not found: {audio_path}")
    #         return


    # def on_share_audio_stop_send_callback(self):
    #     print("on_share_audio_stop_send_callback called")
    #     self.share_audio_sender = None
    

    def write_to_file(self, path, data):
        meeting_event_log.append({
            "event": "write_to_file",
            "path": str(path),
            "ts": datetime.now().strftime("%Y.%m.%d %H:%M:%S.%f")
        })
        try:
            buffer_bytes = data.GetBuffer()

            with open(path, 'ab') as file:
                file.write(buffer_bytes)
        except IOError as e:
            print(f"Error: failed to open or write to audio file path: {path}. Error: {e}")
            return
        except Exception as e:
            print(f"Unexpected error occurred: {e}")
            return


    def start_raw_recording(self):
        self.recording_ctrl = self.meeting_service.GetMeetingRecordingController()

        can_start_recording_result = self.recording_ctrl.CanStartRawRecording()
        if can_start_recording_result != zoom.SDKERR_SUCCESS:
            self.recording_ctrl.RequestLocalRecordingPrivilege()
            print("Requesting recording privilege.")
            return

        start_raw_recording_result = self.recording_ctrl.StartRawRecording()
        if start_raw_recording_result != zoom.SDKERR_SUCCESS:
            print("Start raw recording failed.")
            return
        # --- create output dir & open WAV(s) ---------------------------
        out_dir = pathlib.Path("sample_program/out/audio")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        wav_path = out_dir / f"meeting_{ts}.wav"
        meeting_event_log.append({
            "event": "start_raw_recording",
            "wav_path": str(wav_path),
            "ts": datetime.now().strftime("%Y.%m.%d %H:%M:%S.%f")
        })
        self.mix_wav = wave.open(str(wav_path), "wb")     # ВАЖНО: именно wave.open, не Path
        # self.mix_wav = wave.open(out_dir / f"meeting_{ts}.wav", "wb")
        self.mix_wav.setnchannels(1)       # mono
        self.mix_wav.setsampwidth(2)       # 16-bit PCM
        self.mix_wav.setframerate(32000)   # Zoom SDK default:contentReference[oaicite:0]{index=0}

        self.audio_helper = zoom.GetAudioRawdataHelper()
        if self.audio_helper is None:
            print("audio_helper is None")
            return
        
        if self.audio_source is None:
            self.audio_source = zoom.ZoomSDKAudioRawDataDelegateCallbacks(onOneWayAudioRawDataReceivedCallback=self.on_one_way_audio_raw_data_received_callback, collectPerformanceData=True)

        audio_helper_subscribe_result = self.audio_helper.subscribe(self.audio_source, False)
        print("audio_helper_subscribe_result =",audio_helper_subscribe_result)

        self.virtual_audio_mic_event_passthrough = zoom.ZoomSDKVirtualAudioMicEventCallbacks(onMicInitializeCallback=self.on_mic_initialize_callback,onMicStartSendCallback=self.on_mic_start_send_callback)
        audio_helper_set_external_audio_source_result = self.audio_helper.setExternalAudioSource(self.virtual_audio_mic_event_passthrough)
        print("audio_helper_set_external_audio_source_result =", audio_helper_set_external_audio_source_result)


    def stop_raw_recording(self):
        if self.mix_wav:
            self.mix_wav.close()
            self.mix_wav = None

        for wav in self.user_wavs.values():
            wav.close()
        self.user_wavs.clear()

        rec_ctrl = self.meeting_service.StopRawRecording()
        if rec_ctrl.StopRawRecording() != zoom.SDKERR_SUCCESS:
            raise RuntimeError("Error with stop raw recording")


    def leave(self):
        if self.meeting_service is None:
            return
        
        status = self.meeting_service.GetMeetingStatus()
        if status == zoom.MEETING_STATUS_IDLE:
            return

        self.meeting_service.Leave(zoom.LEAVE_MEETING)
        # self.save_meeting_log("leave")


    def save_meeting_log(self, status: str = ""):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        meeting_log_name = f"log_{self.meeting_id}_{status}_{ts}.json"
        out_dir = pathlib.Path(f"sample_program/out/audio/{self.meeting_name}/{meeting_log_name}")
        with open(out_dir, "w") as f:
            f.write(json.dumps(meeting_event_log))


    def join_meeting(self, user_logged_in=False):
        mid = os.environ.get('MEETING_ID')
        password = os.environ.get('MEETING_PWD')
        display_name = "Честный ассистент"

        meeting_number = int(mid)
        zak = get_zak(self.account_id, self.client_id, self.client_secret)

        join_param = zoom.JoinParam()
        join_param.userType = (zoom.SDK_UT_WITH_LOGIN
                                if user_logged_in else
                                zoom.SDK_UT_WITHOUT_LOGIN)

        param = join_param.param
        param.meetingNumber = meeting_number
        param.userName = display_name
        # param.userId = USER_ID
        param.userZAK       = zak             # ← ключевая строка
        param.psw = password
        param.isVideoOff = False
        param.isAudioOff = False
        param.isAudioRawDataStereo = False
        param.isMyVoiceInMix = False
        param.eAudioRawdataSamplingRate = zoom.AudioRawdataSamplingRate.AudioRawdataSamplingRate_32K

        # param.userZAK = zak
        # param.userId = self.user_id
        param.psw = password

        join_result = self.meeting_service.Join(join_param)
        print("join_result =", join_result)

        self.audio_settings = self.setting_service.GetAudioSettings()
        self.audio_settings.EnableAutoJoinAudio(True)


    def on_reminder_notify(self, content, handler):
        if handler:
            handler.accept()


    def auth_return(self, result):
        if result == zoom.AUTHRET_SUCCESS:
            print("Auth completed successfully.")
            # self.start_login()
            self.join_meeting()
            return 
        raise RuntimeError("Failed to authorize. result =", result)


    def meeting_status_changed(self, status, iResult):
        if status == zoom.MEETING_STATUS_INMEETING:
            return self.on_join()
        print("meeting_status_changed called. status =", status, "iResult=", iResult)
        if status == zoom.MEETING_STATUS_ENDED:
            self.save_meeting_log("meeting_status_changed")
            sys.exit()


    def create_services(self):
        self.meeting_service = zoom.CreateMeetingService()
        self.setting_service = zoom.CreateSettingService()
        self.meeting_service_event = zoom.MeetingServiceEventCallbacks(onMeetingStatusChangedCallback=self.meeting_status_changed)

        meeting_service_set_revent_result = self.meeting_service.SetEvent(self.meeting_service_event)
        if meeting_service_set_revent_result != zoom.SDKERR_SUCCESS:
            raise RuntimeError("Meeting Service set event failed")

        self.auth_event = zoom.AuthServiceEventCallbacks(onAuthenticationReturnCallback=self.auth_return)
        self.auth_service = zoom.CreateAuthService()

        set_event_result = self.auth_service.SetEvent(self.auth_event)
        print("set_event_result =", set_event_result)

        # Use the auth service
        auth_context = zoom.AuthContext()
        auth_context.jwt_token = generate_jwt(os.environ.get('ZOOM_APP_CLIENT_ID'), os.environ.get('ZOOM_APP_CLIENT_SECRET'))
        result = self.auth_service.SDKAuth(auth_context)

        if result == zoom.SDKError.SDKERR_SUCCESS:
            print("Authentication successful")
        else:
            print("Authentication failed with error:", result)
