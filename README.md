# face-recognition
Face Recognition (PCA y KPCA en Pyhton) - Métodos Numéricos Avanzados 2020 - Instituto Tecnológico de Buenos Aires

### Requisitos
Contar con Python 3 instalado.

Contar con las librerías numpy, sklearn, matplotlib y opencv-python instaladas. De no ser así, ejecutar los comandos:
```
pip install [numpy | sklearn | matplotlib]
pip3 install opencv-python
```
Estos comandos puede variar para distintos Sistemas Operativos

### Ejecución

Para ejecutar el programa se debe elegir el método a utilizar (PCA o KPCA), que se pasará por parámetro. 

Luego, desde la carpeta del proyecto, ejecutar el comando:
```
python3 main.py [pca | kpca]
```
Una vez que el programa esté en ejecución, se encenderá la cámara web de su computadora y deberá presionar "r" para tomar la fotografía que el programa analizará.

###Referencias

####Teoría
[1] https://towardsdatascience.com/building-a-face-recognizer-in-python-7fd6630c6340 \
[2] https://towardsdatascience.com/principal-component-analysis-algorithm-in-real-life-discovering-patterns-in-a-real-estate-dataset-18134c57ffe7 \
[3] http://staff.ustc.edu.cn/~zwp/teach/MVA/pcaface.pdf \
[4] https://es.wikipedia.org/wiki/M%C3%A1quinas_de_vectores_de_soporte \

####OpenCV
[5] https://github.com/opencv/
