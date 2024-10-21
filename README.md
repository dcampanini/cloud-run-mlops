# Código para generar una google cloud run capaz de simular un ciclo de MLOps

El presente código permite monitorear la creación de archivos en un bucket storage en particular,
tal que cada vez que llega un nuevo archivo al bucket la cloud run se ejecuta y se entrena nuevamente 
el modelo que se especifica en el código.

El algoritmo utilizado es un XGboost y el dataset de entrada es un csv, más detalles del problema abordado
se puede encontrar en el siguiente link: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data

Para el correcto funcionamiento de la cloud run se espera que el archivo que se crea en el bucket sea el provisto 
en el presente repositorio y que lleva por nombre pima-indians-diabetes.csv

Al finalizar la ejecuci+on de la  cloud run esta almacena el modelo entrenado en formato pkl en un bucket storage