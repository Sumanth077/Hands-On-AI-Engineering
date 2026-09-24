import os

from dotenv import load_dotenv

from ui.gradio_app import build_app

load_dotenv()

if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

    app = build_app()
    app.launch(server_name=server_name, server_port=server_port)
