import streamlit as st
import subprocess
import os
import sys


# --- Helper Functions ---

def is_tool_installed(name):
    """Check whether `name` is on PATH and marked as executable."""
    from shutil import which
    return which(name) is not None


def get_video_title(url):
    """Gets the video title using yt-dlp."""
    try:
        command = [sys.executable, "-m", "yt_dlp", "--get-title", "--no-warnings", url]
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        st.error(f"Error fetching video title: {e.stderr}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None


def download_audio(url, output_filename="audio.mp3"):
    """Downloads audio from a YouTube URL and converts it to MP3."""
    try:
        command = [
            sys.executable, "-m", "yt_dlp",
            "-x",  # Extract audio
            "--audio-format", "mp3",
            "--no-warnings",
            "-o", output_filename,
            url
        ]
        with st.spinner('Downloading and converting video... Please wait.'):
            # We use Popen to potentially handle long-running downloads better in future versions
            # For now, we wait for it to complete.
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                st.error(f"Download failed. Error: {stderr.decode('utf-8', 'ignore')}")
                return False
        return True
    except Exception as e:
        st.error(f"An error occurred during the download process: {e}")
        return False


# --- Streamlit App UI ---

st.set_page_config(page_title="YouTube to MP3", page_icon="🎵", layout="centered")

st.title("🎵 YouTube to MP3 Converter")
st.markdown("Enter a YouTube video URL below to download its audio as a high-quality MP3 file.")

# Check for the ffmpeg dependency, which is crucial for audio extraction.
if not is_tool_installed("ffmpeg"):
    st.error("🔴 **Error: `ffmpeg` is not installed or not in your system's PATH.**")
    st.markdown("""
        `ffmpeg` is a required dependency for converting and extracting audio. Please install it to use this app.

        **Installation Instructions:**
        - **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin` folder to your system's PATH.
        - **macOS (using Homebrew):** Run `brew install ffmpeg` in your terminal.
        - **Debian/Ubuntu Linux:** Run `sudo apt-get update && sudo apt-get install ffmpeg` in your terminal.
    """)
else:
    # Input field for the YouTube URL
    youtube_url = st.text_input(
        "Enter YouTube URL:",
        placeholder="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    )

    if st.button("Convert to MP3", type="primary"):
        if youtube_url:
            # Sanitize the URL a bit
            cleaned_url = youtube_url.split('&')[0]

            # Define a temporary, unique filename for the output
            output_filename = "downloaded_audio.mp3"

            # Clean up any previous downloads before starting
            if os.path.exists(output_filename):
                os.remove(output_filename)

            video_title = get_video_title(cleaned_url)

            if video_title:
                st.write(f"**▶️ Video Found:** *{video_title}*")
                if download_audio(cleaned_url, output_filename):
                    st.success("✅ Conversion successful!")

                    # Sanitize title for a valid filename
                    safe_filename = "".join(
                        [c for c in video_title if c.isalpha() or c.isdigit() or c in ' ._-']).rstrip()
                    final_filename = f"{safe_filename}.mp3"

                    try:
                        with open(output_filename, "rb") as file:
                            st.download_button(
                                label=f"Click to Download '{final_filename}'",
                                data=file,
                                file_name=final_filename,
                                mime="audio/mpeg"
                            )
                        # Clean up the file after making it available for download
                        if os.path.exists(output_filename):
                            os.remove(output_filename)
                    except FileNotFoundError:
                        st.error("Error: The converted file could not be found. Please try again.")
        else:
            st.warning("Please enter a valid YouTube URL to begin.")

# Footer
st.markdown("---")
st.markdown(
    "Built with ❤️ using [Streamlit](https://streamlit.io) and powered by [yt-dlp](https://github.com/yt-dlp/yt-dlp).")
