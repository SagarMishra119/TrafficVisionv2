def image_uploader(st, key: str = "img_upload"):
    return st.file_uploader("Upload traffic image", type=["jpg", "jpeg", "png"], key=key)


def video_uploader(st, key: str = "vid_upload"):
    return st.file_uploader("Upload traffic video", type=["mp4", "avi", "mov"], key=key)
