from __future__ import annotations

from pathlib import Path

from IPython.display import Javascript, display
from google.colab import output


def take_photo(
    filename: str | Path = "camera.jpg",
    quality: float = 0.8,
    width: int = 640,
    height: int = 480,
) -> str:
    """Open the browser camera in Colab and save one captured frame."""

    filename = str(filename)
    display(
        Javascript(
            f"""
            async function takePhoto(quality, width, height) {{
              const div = document.createElement('div');
              const video = document.createElement('video');
              const button = document.createElement('button');
              button.textContent = '撮影';
              button.style.display = 'block';
              button.style.margin = '12px 0';

              div.appendChild(video);
              div.appendChild(button);
              document.body.appendChild(div);

              const stream = await navigator.mediaDevices.getUserMedia({{
                video: {{ width: width, height: height }}
              }});
              video.srcObject = stream;
              await video.play();

              google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

              await new Promise((resolve) => button.onclick = resolve);

              const canvas = document.createElement('canvas');
              canvas.width = video.videoWidth;
              canvas.height = video.videoHeight;
              canvas.getContext('2d').drawImage(video, 0, 0);

              stream.getVideoTracks()[0].stop();
              div.remove();

              return canvas.toDataURL('image/jpeg', quality);
            }}
            """
        )
    )

    data_url = output.eval_js(f"takePhoto({quality}, {width}, {height})")
    binary = _data_url_to_bytes(data_url)
    Path(filename).write_bytes(binary)
    return filename


def _data_url_to_bytes(data_url: str) -> bytes:
    import base64

    header, encoded = data_url.split(",", 1)
    if not header.startswith("data:image/"):
        raise ValueError("Captured data is not an image.")
    return base64.b64decode(encoded)
