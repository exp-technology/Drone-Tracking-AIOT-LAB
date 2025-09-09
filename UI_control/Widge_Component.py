from PyQt6 import QtWidgets


class Label_Component(QtWidgets.QLabel):
    def __init__(self, text, size=12):
        super().__init__()
        self.setText(text)
        self.setStyleSheet(f"font-family: Arial; font-size: {size}px; font-weight: bold;")


class Button_Component(QtWidgets.QPushButton):

    def __init__(self, text, text_size=12, size=(150, 30),
                 color_bk=(69, 140, 225), color_hover=(163, 220, 251), color_pressed=(115, 148, 166)):
        super().__init__()
        self.text_size = text_size
        self.color_bk = color_bk
        self.color_hover = color_hover
        self.color_pressed = color_pressed

        self.setText(text)
        self.setFixedSize(size[0], size[1])
        self.setStyleSheet("QPushButton{"
                           f"   border:1px solid; border-radius:10px;"
                           f"   font-family: Arial; font-size:{text_size}pt; font-weight:bold;"
                           f"   background-color: rgb({color_bk[0]}, {color_bk[1]}, {color_bk[2]});"
                           "}"
                           "QPushButton:hover{"
                           f"   background-color: rgb({color_hover[0]}, {color_hover[1]}, {color_hover[2]});"
                           "}"
                           "QPushButton:pressed{"
                           f"    background-color: rgb({color_pressed[0]}, {color_pressed[1]}, {color_pressed[2]});"
                           "    border-style: inset;}")

    def press_event(self):
        self.setStyleSheet("QPushButton{"
                           f"   border:1px solid; border-radius:10px;"
                           f"   font-family: Arial; font-size:{self.text_size}pt; font-weight:bold;"
                           f"   background-color: rgb({self.color_pressed[0]}, {self.color_pressed[1]}, {self.color_pressed[2]});"
                           "}")

    def release_event(self):
        self.setStyleSheet("QPushButton{"
                           f"   border:1px solid; border-radius:10px;"
                           f"   font-family: Arial; font-size:{self.text_size}pt; font-weight:bold;"
                           f"   background-color: rgb({self.color_bk[0]}, {self.color_bk[1]}, {self.color_bk[2]});"
                           "}"
                           "QPushButton:hover{"
                           f"   background-color: rgb({self.color_hover[0]}, {self.color_hover[1]}, {self.color_hover[2]});"
                           "}"
                           "QPushButton:pressed{"
                           f"    background-color: rgb({self.color_pressed[0]}, {self.color_pressed[1]}, {self.color_pressed[2]});"
                           "    border-style: inset;}")

