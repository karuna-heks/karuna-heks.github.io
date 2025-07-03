MyPrivateRecorder is a lightweight personal utility that automatically joins Zoom meetings, records the mixed audio stream locally (WAV) and optionally sends it to a speech-to-text engine running on the same computer. The app never stores data on external servers and does not access video or participant lists beyond what is needed to capture the in-meeting audio.


### Installing MyPrivateRecorder
1. Click **Add to Zoom** on this page.
2. Grant the requested permission (join audio) and finish installation.

### Using
1. Start your Zoom meeting.
2. Launch the recorder (CLI: `zoom-recorder --join <meeting-id> --pwd <pass>`).
3. When the meeting ends an audio file appears in the configured folder.

### Removing
Open Zoom “Apps → Manage” → click “Remove” next to **MyPrivateRecorder**.
