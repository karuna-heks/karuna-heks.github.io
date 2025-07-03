import os
import signal
import argparse
from typing import Optional, Tuple
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
import gi
gi.require_version('GLib', '2.0')
from gi.repository import GLib

load_dotenv()

def parse_zoom_link(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Извлекает ID конференции Zoom и секрет (pwd) из переданной ссылки.

    Параметры
    ----------
    url : str
        Ссылка вида https://…zoom.us/j/<MEETING_ID>?pwd=<SECRET>

    Возвращает
    ----------
    Tuple[str | None, str | None]
        (meeting_id, secret).  Если что-то не найдено, на этом месте будет None.
    """
    parsed = urlparse(url)

    # ---------- ID конференции ----------
    meeting_id = None
    # Преобразуем путь '/j/83053648874' → ['j', '83053648874']
    parts = [p for p in parsed.path.split('/') if p]
    if 'j' in parts:                           # классический формат /j/<ID>
        j_idx = parts.index('j')
        if len(parts) > j_idx + 1:
            meeting_id = parts[j_idx + 1]
    elif parts and parts[0].isdigit():         # reserve: /<ID> без /j/
        meeting_id = parts[0]

    # ---------- Секрет (pwd) ----------
    query = parse_qs(parsed.query)
    secret = query.get('pwd', [None])[0]       # в большинстве случаев
    # иногда встречается ?passcode=…
    if secret is None:
        secret = query.get('passcode', [None])[0]

    return meeting_id, secret


class ZoomBotRunner:
    def __init__(self, meeting_id: str, secret: str):
        self.bot = None
        self.main_loop = None
        self.shutdown_requested = False
        self.meeting_id = meeting_id
        self.secret = secret

    def exit_process(self):
        """Clean shutdown of the bot and main loop"""
        print("Starting cleanup process...")

        # Set flag to prevent re-entry
        if self.shutdown_requested:
            return False
        self.shutdown_requested = True

        try:
            if self.bot:
                print("Leaving meeting...")
                self.bot.leave()
                print("Cleaning up bot...")
                self.bot.cleanup()
                self.force_exit()

        except Exception as e:
            print(f"Error during cleanup: {e}")
            self.force_exit()
        return False

    def force_exit(self):
        """Force the process to exit"""
        print("Forcing exit...")
        os._exit(0)  # Use os._exit() to force immediate termination
        return False

    def on_signal(self, signum, frame):
        """Signal handler for SIGINT and SIGTERM"""
        print(f"\nReceived signal {signum}")
        # Schedule the exit process to run soon, but not immediately
        if self.main_loop:
            GLib.timeout_add(100, self.exit_process)
        else:
            self.exit_process()

    def on_timeout(self):
        """Regular timeout callback"""
        if self.shutdown_requested:
            return False
        return True

    def run(self):
        """Main run method"""
        os.environ['MEETING_ID'] = self.meeting_id.strip()
        os.environ['MEETING_PWD'] = self.secret.strip()
        from meeting_bot import MeetingBot
        self.bot = MeetingBot(self.meeting_id, self.secret)
        try:
            self.bot.init()
        except Exception as e:
            print(e)
            self.exit_process()

        # Create a GLib main loop
        self.main_loop = GLib.MainLoop()

        # Add a timeout function that will be called every 100ms
        GLib.timeout_add(100, self.on_timeout)

        try:
            print("Starting main event loop")
            self.main_loop.run()
        except KeyboardInterrupt:
            print("Interrupted by user, shutting down...")
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.exit_process()


def main() -> None:

    
    parser = argparse.ArgumentParser(
        description="Запускает бота для указанной Zoom-конференции")
    parser.add_argument(
        "--zoom_url", "-z", default=None,
        help="Ссылка на Zoom-встречу (в кавычках, чтобы не «сломать» &)"
    )
    parser.add_argument(
        "--meeting_id", "-m", default=None,
        help="meetingID:meetingPass"
    )
    args = parser.parse_args()
    if args.zoom_url is not None:
        meeting_id, secret = parse_zoom_link(args.zoom_url)
        meeting_id = meeting_id.strip()
        secret = secret.strip()
    elif args.meeting_id is not None:
        meeting_id, secret = args.meeting_id.split(":")
    else:
        raise RuntimeError("Unknown either zoom_url and meeting_id")
    if not meeting_id or not secret:
        parser.error("Не удалось извлечь meeting_id или secret из переданной ссылки")
    runner = ZoomBotRunner(meeting_id=meeting_id, secret=secret)

    # Set up signal handlers
    signal.signal(signal.SIGINT, runner.on_signal)
    signal.signal(signal.SIGTERM, runner.on_signal)

    # Run the Meeting Bot
    runner.run()

if __name__ == "__main__":
    main()
