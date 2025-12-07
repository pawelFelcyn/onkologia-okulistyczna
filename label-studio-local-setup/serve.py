import os
import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import time
import traceback


RESTART_DELAY = 2


class SafeCORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "*")
        return super().end_headers()

    def do_GET(self):
        try:
            super().do_GET()
        except Exception as e:
            print(f"Error serving {self.path}: {e}")
            traceback.print_exc()


def run_server(serve_dir: str, port: int):
    os.chdir(serve_dir)
    print(f"🚀 Serving directory: {serve_dir} on port {port}")

    server = ThreadingHTTPServer(("0.0.0.0", port), SafeCORSRequestHandler)
    print(f"Server running at http://localhost:{port}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server (Ctrl+C)...")
        server.shutdown()
        print("Server stopped cleanly.")
        raise KeyboardInterrupt
    except Exception as e:
        print(f"Server crashed: {e}")
        traceback.print_exc()
        server.shutdown()
        raise

def normalize_path(p: str) -> str:
    p = p.strip().strip('"').strip("'")
    p = p.replace("\\", "/")
    p = os.path.expanduser(p)
    p = os.path.abspath(p)

    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple CORS-enabled file server with auto-restart."
    )
    # this need to be full path to directory
    parser.add_argument(
        "--serve_dir",
        type=str,
        required=True,
        help="Directory to be served over HTTP"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to serve on (default: 8000)"
    )

    args = parser.parse_args()
    serve_dir = normalize_path(args.serve_dir)
    port = args.port
    if not os.path.isdir(serve_dir):
        raise NotADirectoryError(f"Provided serve_dir does not exist: {serve_dir}")

    while True:
        try:
            run_server(serve_dir, port)
        except KeyboardInterrupt:
            print("Server stopped by user (Ctrl+C). Exiting.")
            break
        except Exception:
            print(f"Restarting server in {RESTART_DELAY} seconds...")
            time.sleep(RESTART_DELAY)
