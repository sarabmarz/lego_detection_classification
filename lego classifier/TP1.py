import cv2 as cv
import numpy as np
from aux_func import psColor, bwLabel
import os, math


def dict_to_string_list(dict_to_convert):
    return str(dict_to_convert).replace('\'', '').replace('{', '').replace('}', '').split(', ')

def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calc_area(min_size, ratio, tolerance):
    return np.power(min_size, 2) * ratio * tolerance

class Identify_Lego_TP1:
    piece_ratios = {
        "2x2": 1.0,   # 1:1
        "2x4": 2.0,   # 2:1
        "2x6": 3.0,   # 3:1
        "2x8": 4.0    # 4:1
    }

    piece_counter = {"2x2 brick count": 0,
                     "2x4 brick count": 0,
                     "2x6 brick count": 0,
                     "2x8 brick count": 0}

    def __init__(self, file_dir, file_name, ipynb=False, min_lego_size_h=85, tolerance=0.25, show_summary=True, verbose=False):
        self.file_dir = file_dir
        self.file_name = file_name
        self.min_size_h = min_lego_size_h
        self.tolerance = tolerance
        self.ipynb = ipynb
        self.show_summary = show_summary
        self.verbose = verbose

        self.load_image()

    def display_image(self, title, image):
        """ Function to display images with OpenCV """
        if self.verbose: 
            cv.imshow(title, image)

    # Step 1: Leitura de Imagens
    def load_image(self):
        """ Load the image and initialize basic parameters """
        self.img = cv.imread(os.path.join(self.file_dir, self.file_name))
        if self.img is None:
            raise ValueError(f"Image not found or unable to load: {self.file_dir + self.file_name}")

        self.img_gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        self.img_gray = cv.GaussianBlur(self.img_gray, (5, 5), 0)
        self.img_height, self.img_width = self.img.shape[:2]

    def mask_color(self):
        chromakey_color = np.array([50, 139, 218])
        tolerance = 5

        mask = cv.inRange(self.img, chromakey_color - tolerance, chromakey_color + tolerance)
        mask_inv = 255 - mask

        maskImage = cv.cvtColor(mask_inv, cv.COLOR_GRAY2BGR)
        img1 = np.uint8(self.img * (maskImage * 1.0 / 255))
        return img1

    def get_background_mask(self):
        # Get Background Mask
        low_blue_bg = np.array([0, 0, 0])
        high_blue_bg = np.array([255, 190, 90])
        bg_mask = cv.inRange(self.img, low_blue_bg, high_blue_bg)
        bg_mask_inv = cv.bitwise_not(bg_mask)
        
        return bg_mask_inv
    
    def morph_from_mask(self):
        return
    
    # Step 2: Binarização
    def binarize_image(self):
        """ Split the image and binarize the red channel """

        _, img_green, img_red =  cv.split(self.img)
        combined = cv.addWeighted(img_red, .5, img_green, .5, 0)
        combined = cv.GaussianBlur(combined, (5, 5), 0)

        str_elemr, bwr = self.get_morph_from_img(combined, thresh=127, thresh_type=cv.THRESH_OTSU)


        self.display_image('Combined Red and Green Channels', combined)
        self.display_image('bwr', bwr)

        return str_elemr, bwr

    # Step 3: Melhoramento da Imagem
    def enhance_image(self, bwr, str_elemr):
        """ Apply morphological enhancement to the binary image """
        
        self.enhanced_image = cv.erode(bwr, str_elemr, iterations=1)

        self.display_image('image', self.enhanced_image)
        return self.enhanced_image

    def get_morph_from_img(self, image, mask=None, thresh=150, thresh_type=cv.ADAPTIVE_THRESH_GAUSSIAN_C, struct_size=3):
        """ Morphological processing """
        if mask is not None:
            image = image * mask

        _, bw = cv.threshold(image, thresh, 255, thresh_type)
        self.display_image('just thresh bw', bw)
        ksize = (struct_size, struct_size)
        str_elem = cv.getStructuringElement(cv.MORPH_RECT, ksize)
        bw_morph = cv.morphologyEx(bw, cv.MORPH_CLOSE, str_elem, iterations=1)
    
        if thresh_type == cv.ADAPTIVE_THRESH_GAUSSIAN_C:
            bw_morph = cv.bitwise_not(bw_morph)
        return str_elem, bw_morph

    # Step 4: Classificação das Peças
    def handle_border_touch(self, contour, w, h):
        """ Check if the piece touches the border and adjust height if needed """
        if self.does_touch_border(contour, w, h) and h < self.min_size_h:
            h = self.min_size_h
            return True, h
        return False, h
    
    def is_rectangle(self, contour):
        """ Check if a contour is rectangular based on aspect ratio and vertices. """
        epsilon = 0.03
        perimeter = cv.arcLength(contour, True) 
        # Fração do comprimento total da borda e controla o quanto a forma do contorno pode ser simplificada
        approx = cv.approxPolyDP(contour, epsilon * perimeter, True)
        
        print("Vertex Number:", len(approx))

        cv.polylines(self.img, approx, True, [255, 0, 255], 3)

        return len(approx) == 4

    def detect_and_split_merged_pieces(self, contour):
        hull = cv.convexHull(contour, returnPoints=False)
        hull[::-1].sort(axis=0)
        defects = cv.convexityDefects(contour, hull)

        defect_points = []
        thresh_mod = 15
        
        if defects is not None:
            d_values = [defects[i, 0][3] for i in range(defects.shape[0])]
            threshold = np.median(d_values) * thresh_mod
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                # start = tuple(contour[s][0])
                # end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                # Confirmar na blibliografia exatamente o que isto está a fazer
                
                if d > threshold: 
                    cv.circle(self.img, far, 3, [0, 0, 255], -1)
                    defect_points.append(far)
                # else:
                #     cv.circle(self.img, far, 3, [0, 255, 255], -1)

            # Draw a line between two closest points
            while len(defect_points) > 1:
                min_distance = 15
                
                for i in range(len(defect_points)):
                    for j in range(i + 1, len(defect_points)):
                        point1, point2 = defect_points[i], defect_points[j]
                        dist = distance(point1, point2)
                        print(f'{dist = }')

                        if dist < min_distance:
                            min_distance = dist
                            cv.line(self.enhanced_image, point1, point2, 0, 3)
                            self.display_image('LINES?', self.enhanced_image)
                
                # Remove that pair
                defect_points.remove(point1)
                defect_points.remove(point2)

        # epsilon = 0.03 * cv.arcLength(contour, True)
        # approx = cv.approxPolyDP(contour, epsilon, True)

        return contour

    def get_piece_type(self, contour, aspect_ratio):
        """ Determine the closest piece type based on aspect ratio """
        closest_piece = "Unknown"
        for piece, ratio in self.piece_ratios.items():
            if (abs(aspect_ratio - ratio) < self.tolerance
                and self.is_rectangle(contour)
                and self):
                closest_piece = piece
                break
        if closest_piece == "Unknown":
            print("ABOVE IS UNKNOWN!")
        return closest_piece

    def get_min_max_ratios(self):
        """ Get the smallest and largest ratios from the piece_ratios dictionary """
        min_ratio = min(self.piece_ratios.values())
        max_ratio = max(self.piece_ratios.values())
        return min_ratio, max_ratio

    def check_valid_area(self, contour):
        """ Check if the area of the piece is within allowed bounds based on piece_type """

        area = round(cv.contourArea(contour))


        min_ratio, max_ratio = self.get_min_max_ratios()

        min_area = calc_area(self.min_size_h, min_ratio, 1 - self.tolerance)
        max_area = calc_area(self.min_size_h, max_ratio, 1 + self.tolerance)

        print(f'{min_area = }')
        print(f'{max_area = }')

        area_valid = min_area <= area <= max_area

        return area_valid

    def classify_piece(self, contour):
        """ Classify each detected piece based on aspect ratio """

        if not self.is_rectangle(contour):
              contour = self.detect_and_split_merged_pieces(contour)

        if not self.check_valid_area(contour):
            print('WRONG PIECE')
            return None
        
        area = cv.contourArea(contour)

        rect = cv.minAreaRect(contour)
        (x, y), (w, h), angle = rect

        if w < h:
            w, h = h, w

        if w == 0 or h == 0:
            return None
        
        #self.is_rectangle(contour)
        border_touch, adjusted_height = self.handle_border_touch(contour, w, h)

        aspect_ratio = w / adjusted_height

        piece_type = self.get_piece_type(contour, aspect_ratio)


        if border_touch:
            piece_type = f"Unknown ({piece_type})"


        return {
            'piece_type': piece_type,
            'aspect_ratio': aspect_ratio,
            'width': w,
            'height': adjusted_height,
            'angle': angle,
            'area': area,
            'center': (int(x), int(y)),
            'contour': np.int32(cv.boxPoints(rect))
        }

    def does_touch_border(self, contour, w, h):
        """ Check if the contour touches the border of the image """
        x_min, y_min, w_rect, h_rect = cv.boundingRect(contour)
        touches_border = (x_min <= 0 
                          or y_min <= 0 
                          or x_min + w_rect >= self.img_width 
                          or y_min + h_rect >= self.img_height)
        return touches_border

    # Step 5: Labeling and Pseudo-coloring the Image
    def label_regions(self, bwr, thresh=127):
        """ Label connected components and create a pseudo-colored image """
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(bwr)
        
        valid_contours = []
        for i in range(1, num_labels):  
            if stats[i, cv.CC_STAT_AREA] > thresh:  # Adjust threshold as needed
                valid_contours.append(labels == i)
        
        color_map = psColor.CreateColorMap(num_labels)
        pseudo_colored_img = psColor.Gray2PseudoColor(labels, color_map)
        
        return num_labels, pseudo_colored_img


    def run(self):
        """ Main pipeline for piece detection and classification """

        # Step 1: Load Image
        self.load_image()

        # Step 2: Binarize
        str_elemr, bwr = self.binarize_image()

        # Step 3: Enhance Image
        enhanced_img = self.enhance_image(bwr, str_elemr)

        # Step 4: Label Regions
        region_num, pseudo_colored_img = self.label_regions(bwr, 120)
        print(f'Number of regions: {region_num}')

        self.display_image('Pseudo-colored Image', pseudo_colored_img)

        # Step 5: Detect and Classify Pieces
        pieces_info = self.detect_and_classify(iterations=3)
        self.reset_pieces_counter()

        # Log and draw the results on the image
        for piece in pieces_info:
            print(f"\nDetected Piece Type: {piece['piece_type']}")
            print(f"Aspect Ratio: {piece['aspect_ratio']}")
            print(f"Width: {piece['width']}, Height: {piece['height']}, Angle: {piece['angle']}, Area: {piece['area']}")

            # Draw the contour and label the piece
            cv.drawContours(self.img, [piece['contour']], 0, (0, 255, 0), 2)
            cv.putText(self.img, piece['piece_type'], piece['center'], cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Add to counter
            if piece['piece_type'].split(' ')[0] == "Unknown":
                continue
            self.piece_counter[f"{piece['piece_type']} brick count"] += 1

        # print(f'{self.piece_counter = }')
        # print(f'{self.piece_counter.items() = }')
        # print(f'{self.piece_counter.keys() = }')
        
        self.write_on_image(self.piece_counter) if self.show_summary else None

        # Display the final image with detected pieces
        self.display_image('Detected Lego Pieces', self.img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def detect_and_classify(self, iterations=1):
        for i in range(iterations):
            contours = self.detect_pieces(self.enhanced_image)
            pieces_info = []
            for contour in contours:
                piece_info = self.classify_piece(contour)
                if piece_info:
                    pieces_info.append(piece_info)
        
        return pieces_info

    def detect_pieces(self, eroded):
        """ Find contours in the eroded binary image """
        contours, _ = cv.findContours(eroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        return contours

    def reset_pieces_counter(self):
        self.piece_counter = {"2x2 brick count": 0,
                              "2x4 brick count": 0,
                              "2x6 brick count": 0,
                              "2x8 brick count": 0}
    
    def write_on_image(self, info, color=(0, 100, 200), scale=.7, position="top-left"):
        items = list(self.piece_counter.items())
        count_mod = 1
        match position:
            case "top-left":
                for l in range(len(items)):
                    label, count = items[l]
                    if count > 0:
                        cv.putText(self.img, label + ': ' + str(count), (0, int(30 * scale * (count_mod + l))), 
                                   cv.FONT_HERSHEY_SIMPLEX, scale, color, 2)
                    else: count_mod -= 1

    def get_pieces_summary(self):
        return dict_to_string_list(self.piece_counter)


file_dir = "./treino/"
myList = os.listdir(file_dir)
file_names = []

for imgs in myList:
    if imgs.endswith((".jpg", ".png")):
        file_names.append(imgs)

for file_name in file_names:
     piece_identifier = Identify_Lego_TP1(file_dir, file_name, verbose=True)
     piece_identifier.run()

# piece_identifier = PieceIdentifier(file_dir, 'lego11.jpg')
# piece_identifier.run()

# problem_files = ['lego02', 'lego04', 'lego09', 'lego10', 'lego13', 'lego21', 'lego30']
# # problem_files = ['lego10', 'lego11']
# for file_name in problem_files:
#      piece_identifier = PieceIdentifier(file_dir, file_name + '.jpg')
#      piece_identifier.run()