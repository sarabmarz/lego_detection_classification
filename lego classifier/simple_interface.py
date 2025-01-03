import os
import cv2
from tkinter import Label, Tk, Button, filedialog, PhotoImage, Frame, StringVar
from TP1 import Identify_Lego_TP1

# Constants
WINDOW_SIZE = "900x600"
WINDOW_TITLE = "PIV Lego Pieces Identification"
INITIAL_IMAGE_FOLDER = "./treino/"
PROCESSED_IMAGE_FOLDER = "./Processed_Lego_Imgs"

class LegoPieceApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry(WINDOW_SIZE)
        self.root.title(WINDOW_TITLE)

        self.image_paths = []
        self.current_image_index = 0
        self.piece_count_text = StringVar()

        self.init_ui()
        self.load_images_from_folder(INITIAL_IMAGE_FOLDER)

    def init_ui(self):
        """Setup the UI components including labels and buttons."""
        self.image_label = Label(self.root, bg="black")
        
        default_bg = self.root.cget("bg")  
        piece_count_frame = Frame(self.root, bg=default_bg)
        piece_count_frame.place(x=680, y=47, width=180, height=110)
        
        piece_count_label = Label(
            piece_count_frame,
            textvariable=self.piece_count_text,
            font=("Arial", 16),
            bg=default_bg,  
            fg="darkblue",
            anchor="nw",
            justify="center"
        )
        piece_count_label.pack(fill="both", expand=True)
        
        Button(self.root, text="◀", bg="gray", fg="white", command=self.show_previous_image).place(x=390, y=500, width=40)
        Button(self.root, text="▶", bg="gray", fg="white", command=self.show_next_image).place(x=490, y=500, width=40)
        Button(self.root, text="🗁", bg="gray", fg="white", command=self.select_folder).place(x=290, y=500, width=40)
        Button(self.root, text="▽", bg="gray", fg="white", command=self.save_processed_images).place(x=190, y=500, width=40)

    def load_images_from_folder(self, folder_path):
        """Load images with .jpg or .png extensions from the specified folder."""
        self.image_paths.clear()
        if os.path.exists(folder_path):
            files = os.listdir(folder_path)
            self.image_paths = [
                os.path.join(folder_path, file)
                for file in files if file.endswith((".jpg", ".png"))
            ]
            if self.image_paths:
                self.root.after(100, lambda: self.display_image(0))  
        else:
            print("Invalid folder path.")


    def process_image(self, img_path):
        """Process an image using PieceIdentifier."""
        piece_identifier = Identify_Lego_TP1(
            file_dir=os.path.dirname(img_path),
            file_name=os.path.basename(img_path),
            show_summary=False, verbose = False
        )
        piece_identifier.run()
        self.update_piece_count(piece_identifier.get_pieces_summary())
        return piece_identifier.img

    def display_image(self, index):
        """Display the image at the specified index with classified LEGO pieces."""
        self.current_image_index = index
        img_path = self.image_paths[self.current_image_index]
        processed_img = self.process_image(img_path)
        if processed_img is not None:
            self.show_resized_image(processed_img)

    def show_resized_image(self, img):
        """Resize and display the image in the main window."""
        img_height, img_width, _ = img.shape
        aspect_ratio = img_width / img_height
        max_width, max_height = 540, 540

        if img_width > max_width or img_height > max_height:
            if (max_width / aspect_ratio) <= max_height:
                new_width, new_height = max_width, int(max_width / aspect_ratio)
            else:
                new_width, new_height = int(max_height * aspect_ratio), max_height
        else:
            new_width, new_height = img_width, img_height

        img_resized = cv2.resize(img, (new_width, new_height))
        img_encoded = cv2.imencode('.png', img_resized)[1].tobytes()
        img_tk = PhotoImage(data=img_encoded)

        center_x = (self.root.winfo_width() - new_width) // 2 - 90
        center_y = (self.root.winfo_height() - new_height) // 2 - 50

        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk
        self.image_label.place(x=center_x, y=center_y, width=new_width, height=new_height)

    def update_piece_count(self, piece_summary):
        """Update the piece count display with the latest piece classification summary."""
        valid_pieces = [
            f"{item.split()[0]} - {item.split()[-1]} piece(s)"
            for item in piece_summary
            if int(item.split()[-1]) > 0 
        ]

        formatted_summary = "\n".join(valid_pieces) if valid_pieces else "No Valid Piece(s)"
        self.piece_count_text.set(formatted_summary)

    def show_next_image(self):
        """Show the next image in the list."""
        next_index = (self.current_image_index + 1) % len(self.image_paths)
        self.display_image(next_index)

    def show_previous_image(self):
        """Show the previous image in the list."""
        previous_index = (self.current_image_index - 1) % len(self.image_paths)
        self.display_image(previous_index)

    def select_folder(self):
        """Prompt user to select a folder and load images from it."""
        selected_folder = filedialog.askdirectory()
        if selected_folder:
            self.load_images_from_folder(selected_folder)

    def save_processed_images(self):
        """Process and save all images in image_paths."""
        os.makedirs(PROCESSED_IMAGE_FOLDER, exist_ok=True)
        for img_path in self.image_paths:
            processed_img = self.process_image(img_path)
            if processed_img is not None:
                save_path = os.path.join(PROCESSED_IMAGE_FOLDER, os.path.basename(img_path))
                cv2.imwrite(save_path, processed_img)
        print(f"All images saved to {PROCESSED_IMAGE_FOLDER}.")


if __name__ == "__main__":
    root = Tk()
    app = LegoPieceApp(root)
    root.mainloop()
