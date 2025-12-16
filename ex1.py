import streamlit as st
import numpy as np
from PIL import Image

# =========================================================
#  TRANSFORM FUNCTION (FIXED)
# =========================================================
def apply_transform(img_array, matrix):
    h, w, _ = img_array.shape
    
    # Create output image with proper size
    new_img = np.zeros_like(img_array)
    
    # Iterate through each pixel in the original image
    for i in range(h):
        for j in range(w):
            # Apply transformation: [i, j, 1]
            point = np.array([i, j, 1])
            transformed = matrix @ point
            
            # Get new coordinates
            new_i = int(round(transformed[0]))
            new_j = int(round(transformed[1]))
            
            # Check if new coordinates are within bounds
            if 0 <= new_i < h and 0 <= new_j < w:
                new_img[new_i, new_j] = img_array[i, j]
    
    return new_img

# =========================================================
#  CONVOLUTION FUNCTION
# =========================================================
def apply_convolution(img_array, kernel):
    k = kernel.shape[0]
    pad = k // 2
    padded = np.pad(img_array, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    output = np.zeros_like(img_array, dtype=float)

    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            for c in range(3):
                region = padded[i:i+k, j:j+k, c]
                output[i, j, c] = np.sum(region * kernel)
    
    return np.clip(output, 0, 255).astype(np.uint8)


# =========================================================
#  PAGE 1 – HOME
# =========================================================
def home_page():
    st.title("Matrix Transformations & Image Processing Web App")

    st.header("What Does This App Do?")
    st.write(
        "This application demonstrates how matrices are used in image manipulation, "
        "such as translation, rotation, scaling, shearing, reflection, and convolution-based filters, "
        "such as blur and sharpen."
    )

    st.header("What is Matrix Transformations?")
    st.write(
        "Matrix transformation is a mathematical operation that uses a 3x3 matrix to "
        "move the position of pixels in an image."
    )

    st.subheader("Visual Example of Matrix Transformation")
    st.code(
        """
        Translation:
        [1 0 Tx]
        [0 1 Ty]
        [0 0  1]

        Scaling:
        [Sx 0  0]
        [0 Sy  0]
        [0  0  1]

        Rotation:
        [cosθ -sinθ 0]
        [sinθ  cosθ 0]
        [  0     0 1]
        """,
        language="text",
    )

    st.header("What is Convolution?")
    st.write(
        "Convolution is the process of shifting the kernel (small matrix) throughout the image "
        "to produce blur or sharpen effects."
    )

    st.subheader("Kernel Example")
    st.code(
        """
        Blur Kernel:
        [1/9 1/9 1/9]
        [1/9 1/9 1/9]
        [1/9 1/9 1/9]

        Sharpen Kernel:
        [ 0 -1  0]
        [-1  5 -1]
        [ 0 -1  0]
        """,
        language="text",
    )


# =========================================================
#  PAGE 2 – IMAGE PROCESSING TOOLS
# =========================================================
def tools_page():
    st.title("Image Processing Tools")

    tabs = st.tabs(["Translation", "Scaling", "Rotation", "Shearing", "Reflection", "Blur", "Sharpen"])

    uploaded = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if not uploaded:
        st.info("Upload an image to get started.")
        return

    img = Image.open(uploaded).convert("RGB")
    img_array = np.array(img)

    # 1. TRANSLATION
    with tabs[0]:
        st.subheader("Translation")
        st.write("Move the image in X and Y directions")
        tx = st.slider("Translate X (horizontal)", -200, 200, 50, key="tx")
        ty = st.slider("Translate Y (vertical)", -200, 200, 50, key="ty")
        matrix = np.array([[1, 0, ty], 
                          [0, 1, tx], 
                          [0, 0, 1]], dtype=float)
        result = apply_transform(img_array, matrix)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original", use_container_width=True)
        with col2:
            st.image(result, caption="Translated", use_container_width=True)

    # 2. SCALING
    with tabs[1]:
        st.subheader("Scaling")
        st.write("Resize the image")
        sx = st.slider("Scale X", 0.5, 3.0, 1.5, 0.1, key="sx")
        sy = st.slider("Scale Y", 0.5, 3.0, 1.5, 0.1, key="sy")
        matrix = np.array([[sy, 0, 0], 
                          [0, sx, 0], 
                          [0, 0, 1]], dtype=float)
        result = apply_transform(img_array, matrix)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original", use_container_width=True)
        with col2:
            st.image(result, caption="Scaled", use_container_width=True)

    # 3. ROTATION
    with tabs[2]:
        st.subheader("Rotation")
        st.write("Rotate the image around origin")
        angle = st.slider("Angle (degrees)", -180, 180, 45, key="angle")
        rad = np.deg2rad(angle)
        matrix = np.array([[np.cos(rad), -np.sin(rad), 0],
                          [np.sin(rad),  np.cos(rad), 0],
                          [0, 0, 1]], dtype=float)
        result = apply_transform(img_array, matrix)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original", use_container_width=True)
        with col2:
            st.image(result, caption="Rotated", use_container_width=True)

    # 4. SHEARING
    with tabs[3]:
        st.subheader("Shearing")
        st.write("Slant the image")
        shx = st.slider("Shear X", -1.0, 1.0, 0.3, 0.1, key="shx")
        shy = st.slider("Shear Y", -1.0, 1.0, 0.3, 0.1, key="shy")
        matrix = np.array([[1, shy, 0], 
                          [shx, 1, 0], 
                          [0, 0, 1]], dtype=float)
        result = apply_transform(img_array, matrix)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original", use_container_width=True)
        with col2:
            st.image(result, caption="Sheared", use_container_width=True)

    # 5. REFLECTION
    with tabs[4]:
        st.subheader("Reflection")
        st.write("Flip the image")
        axis = st.selectbox("Axis", ["Horizontal", "Vertical"], key="reflect")
        h, w = img_array.shape[:2]
        if axis == "Horizontal":
            # Flip horizontally (across vertical axis)
            matrix = np.array([[1, 0, 0], 
                              [0, -1, w-1], 
                              [0, 0, 1]], dtype=float)
        else:
            # Flip vertically (across horizontal axis)
            matrix = np.array([[-1, 0, h-1], 
                              [0, 1, 0], 
                              [0, 0, 1]], dtype=float)
        result = apply_transform(img_array, matrix)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original", use_container_width=True)
        with col2:
            st.image(result, caption=f"Reflected {axis}", use_container_width=True)

    # 6. BLUR
    with tabs[5]:
        st.subheader("Blur (Convolution)")
        st.write("Apply blur filter using averaging kernel")
        kernel_size = st.selectbox("Kernel Size", [3, 5, 7], key="blur_size")
        blur_kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        result = apply_convolution(img_array, blur_kernel)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original", use_container_width=True)
        with col2:
            st.image(result, caption="Blurred", use_container_width=True)

    # 7. SHARPEN
    with tabs[6]:
        st.subheader("Sharpen (Convolution)")
        st.write("Enhance edges in the image")
        sharpen_kernel = np.array([[0, -1, 0],
                                   [-1,  5, -1],
                                   [0, -1, 0]], dtype=float)
        result = apply_convolution(img_array, sharpen_kernel)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original", use_container_width=True)
        with col2:
            st.image(result, caption="Sharpened", use_container_width=True)


# =========================================================
#  PAGE 3 – TEAM MEMBERS (DENGAN FOTO PROFIL)
# =========================================================
def team_page():
    st.title("👥 Team Members")
    st.write("Meet our amazing team who worked on this project!")
    
    # Data anggota tim dengan foto profil
    members = {
        "Agni Aisyah Putri": {
            "ID": "004202400137",
            "photo": "agni.jpeg",  # Foto biru
            "contributions": [
                "Coding transformation matrix functions",
                "Implementing core image processing logic",
                "Creating project report and documentation",
                "Testing and debugging all transformations"
            ],
            "role": "Lead Developer & Algorithm Specialist"
        },
        "Andita Nurul Azizah": {
            "ID": "004202400059",  # ID diperbaiki
            "photo": "andita.jpeg",  # Foto merah
            "contributions": [
                "Developing Team Members page with profiles",
                "Implementing user interface design",
                "Creating application structure and layout",
                "Adding photo profile features"
            ],
            "role": "Frontend Developer & UI Designer"
        },
        "Cahyani Dwi Gemawang": {
            "ID": "004202400044",  # ID diperbaiki
            "photo": "cahyani.jpeg",  # Foto hijau
            "contributions": [
                "Coding home page content and explanations",
                "Developing Image Processing Tools interface",
                "Implementing convolution filters (blur & sharpen)",
                "Creating documentation and user guides"
            ],
            "role": "Backend Developer & Documentation"
        },
    }
    
    # Tampilkan semua anggota tim dalam grid
    st.subheader("🌟 Our Team")
    
    # Buat 3 kolom untuk menampilkan semua anggota
    col1, col2, col3 = st.columns(3)
    
    # Anggota 1: Agni Aisyah Putri
    with col1:
        st.markdown("### Agni Aisyah Putri")
        st.image(members["Agni Aisyah Putri"]["photo"], width=180)
        st.markdown(f"**Student ID:** `{members['Agni Aisyah Putri']['ID']}`")
        st.markdown(f"**Role:** {members['Agni Aisyah Putri']['role']}")
        
        with st.expander("View Contributions"):
            for contrib in members["Agni Aisyah Putri"]["contributions"]:
                st.write(f"• {contrib}")
    
    # Anggota 2: Andita Nurul Azizah
    with col2:
        st.markdown("### Andita Nurul Azizah")
        st.image(members["Andita Nurul Azizah"]["photo"], width=180)
        st.markdown(f"**Student ID:** `{members['Andita Nurul Azizah']['ID']}`")
        st.markdown(f"**Role:** {members['Andita Nurul Azizah']['role']}")
        
        with st.expander("View Contributions"):
            for contrib in members["Andita Nurul Azizah"]["contributions"]:
                st.write(f"• {contrib}")
    
    # Anggota 3: Cahyani Dwi Gemawang
    with col3:
        st.markdown("### Cahyani Dwi Gemawang")
        st.image(members["Cahyani Dwi Gemawang"]["photo"], width=180)
        st.markdown(f"**Student ID:** `{members['Cahyani Dwi Gemawang']['ID']}`")
        st.markdown(f"**Role:** {members['Cahyani Dwi Gemawang']['role']}")
        
        with st.expander("View Contributions"):
            for contrib in members["Cahyani Dwi Gemawang"]["contributions"]:
                st.write(f"• {contrib}")
    
    st.markdown("---")
    
    # Detail view dengan dropdown (opsional)
    st.subheader("📋 Member Details")
    
    # Dropdown untuk memilih anggota
    selected_member = st.selectbox(
        "Select a team member to see more details:",
        list(members.keys())
    )
    
    # Tampilkan detail anggota yang dipilih
    if selected_member:
        data = members[selected_member]
        
        col_photo, col_info = st.columns([1, 2])
        
        with col_photo:
            st.image(data["photo"], width=200, caption=selected_member)
        
        with col_info:
            st.markdown(f"### {selected_member}")
            st.markdown(f"**Student ID:** `{data['ID']}`")
            st.markdown(f"**Role:** {data['role']}")
            
            st.markdown("#### Key Contributions:")
            for i, contrib in enumerate(data["contributions"], 1):
                st.markdown(f"{i}. {contrib}")
            
            # Statistik sederhana
            st.markdown("#### Statistics:")
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("Total Contributions", len(data["contributions"]))
            with col_stat2:
                st.metric("Project Role", data["role"].split("&")[0].strip())
    
    # Footer untuk halaman team
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
        <h4>✨ Project Information</h4>
        <p><strong>Project:</strong> Matrix Transformations & Image Processing Web App</p>
        <p><strong>Course:</strong> Computer Programming / Linear Algebra</p>
        <p><strong>Year:</strong> 2024</p>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
#  MAIN APP
# =========================================================

# Sidebar dengan logo/header kecil
st.sidebar.title("🖼️ Image Processor")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigate to:",
    ["🏠 Home", "🛠️ Image Processing Tools", "👥 Team Members"],
    index=0
)

# Informasi tambahan di sidebar
st.sidebar.markdown("---")
st.sidebar.info(
    "This app demonstrates matrix transformations and image processing techniques."
)

# Footer sidebar
st.sidebar.markdown("---")
st.sidebar.caption("© 2024 Team Matrix Transformations")

# Routing halaman
if "🏠 Home" in page:
    home_page()
elif "🛠️ Image Processing Tools" in page:
    tools_page()
elif "👥 Team Members" in page:
    team_page()