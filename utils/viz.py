from PIL import Image, ImageDraw
def draw_rectangles(width, height, rects, path=None):
    img = Image.new('RGB', (width, height), (255,255,255))
    d = ImageDraw.Draw(img)
    for (x,y,w,h) in rects:
        d.rectangle([x,y,x+w,y+h], outline=(0,0,0), width=2)
    if path: img.save(path)
    return img
