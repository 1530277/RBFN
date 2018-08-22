#-*- coding: utf-8 -*-
import sys
import os
import numpy as np
import re
import time
import math as mt
import statistics as stats
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt

#Definición de funciones que se utilizarán en la ejecución del algoritmo
def calculate_new_centroids(centroids,clusters):
    """
    "clusters" es una matriz no cuadrada donde se encuentran los elementos (en ejecución) de cada cluster,
    para calcular los nuevos centroides se tiene que sacar un promedio por cada elemento. Por ejemplo, para esta
    implementación se tomó en cuenta solo 2 dimensiones (x,y), entonces, esta función hace la suma de todos los "x"
    y por separado todos los "y", finalmente divide entre el numero de elementos en el cluster "K" y esos serían
    los elementos para el nuevo centroide "K".
    """
    x_sum,y_sum=0,0
    new_centroids=np.zeros(np.shape(centroids))
    element_shape=np.shape(clusters[0])

    for i in range(0,len(clusters)):
        for element_n in clusters[i]:
            x=np.asarray(element_n)
            y=np.asarray(new_centroids[i])
            new_centroids[i]=x+y

    for i in range(0,len(clusters)):
    	for j in range(0,len(new_centroids[i])):
            new_centroids[i][j]=new_centroids[i][j]/len(clusters[i])
    return  new_centroids

def distances_matrix(points_matrix,centroids):
    """Función que retorna una matriz de K columnas (las columnas representan el numero de centroides requeridos)
    y N elementos/filas, se calcula la distancia euclidiana de cada elemento en points_matrix hacia cada uno de los centroides
    """
    distance_matriz=np.zeros([len(points_matrix),len(centroids)])
    i,j=0,0
    for po_m in points_matrix:
        for cent in centroids:
            var_index_centroid=np.sqrt(np.sum((po_m-cent)**2,axis=None))
            distance_matriz[i,j]=var_index_centroid
            j+=1
        i+=1
        j=0
    return distance_matriz

def olds_vs_news(olds,news):
    """
    Compara la distancia de una matriz de centroides de una iteración anterior y de centroies "recien calculados".
    El valor de comparación puede ser modificable dependiendo el dataset, en este caso se utilizó el valor de 0.001
    para todos los dataset.
    """
    flag=False
    v_flag=np.zeros([len(news)])
    ceros=np.zeros([len(news)])
    if(len(olds)==len(news)):
        for i in range(0,len(olds)):
            distance=np.sqrt(np.sum((olds[i]-news[i])**2,axis=0))
            #print("distance-"+str(i)+": "+str(distance))
            if(distance<0.00000000000000000000000000000000000000000000000000000000000000000001):
                v_flag[i]=1
            #print("suma: "+str(np.sum(v_flag+ceros,axis=0)))
        if(np.sum(v_flag+ceros,axis=0)==len(v_flag)):
            return False
        else:
            return True

def matriz_indices(f,c):
    """
    #Crea matriz de indices dimenciones FxC. Ejemplo, si f=2 y c=3 entonces, matriz_indices=[[0,1,2],[0,1,2]]
    """
    matriz_indices=[]
    for i in range(0,f):
        columna=[]
        for j in range(0,c):
            columna.append(j)
        matriz_indices.append(columna)
    return (np.asarray(matriz_indices))

def sort_index(distances_m,matriz_de_indices):
    #Función que acomoda los indices de la matriz de indices y coloca la distancia mínima de cada fila en la primer columna
    
    """m=0
    for sq_d in distances_m:
        for i in range(0,len(sq_d)):
            for j in range(0,len(sq_d)):
                if(sq_d[i]<sq_d[j] and i!=j):
                    print(str(sq_d[i])+", "+str(sq_d[j]))
                    sq_d_respaldo=sq_d[j]
                    sq_d[j]=sq_d[i]
                    sq_d[i]=sq_d_respaldo
                    sq_d_respaldo=matriz_de_indices[m,j]
                    matriz_de_indices[m,j]=matriz_de_indices[m,i]
                    matriz_de_indices[m,i]=sq_d_respaldo
        m+=1"""
    for k in range(0,len(distances_m)):
        for i in range(0,len(distances_m[k])):
            for j in range(0,len(distances_m[k])):
                if(distances_m[k][i]<distances_m[k][j]):
                    respaldo=distances_m[k][j]
                    distances_m[k][j]=distances_m[k][i]
                    distances_m[k][i]=respaldo
                    sq_d_respaldo=matriz_de_indices[k,j]
                    matriz_de_indices[k,j]=matriz_de_indices[k,i]
                    matriz_de_indices[k,i]=sq_d_respaldo

def iniciar_clusters(data,index_matrix,k):
    clusters=[]
    for i in range(0,k):
        clusters.append(list([]))

    for i in range(0,len(clusters)):
        for j in range(0,len(data)):
            if(index_matrix[j,0]==i):
                clusters[i].append(list(data[j]))
            
    return clusters

def get_n_classes(file_path):
    """
    Obtiene la última columna de cada archivo, donde están las "etiquetas de clase", se hace una lista para después
    borrar los elementos repetidos. La función retorna la cantidad de elementos no repetidos de la lista de etiquetas, así
    se sabe el número de grupos de cada dataset.
    """
    file=open(file_path)
    clases=[]
    flag=0
    index=0
    for row in file:
        row=re.split("\n|,|\t",row)
        if(flag==0):
            length_f=(len(row)-1)
            flag=1
        if(row[length_f]==''):
            row=row[0:length_f]
            length_f-=1    
        clases.append(row[length_f])
    clases=list(set(clases))
    return len(clases)

def get_data(path):
    """
    Lee las primeras 2 columnas del archivo en ejecución
    """
    path_exception1="C:/Users/Luis/Desktop/Examen 2 - Mineria/datasets/wine.txt"
    path_exception2="C:/Users/Luis/Desktop/Examen 2 - Mineria/datasets/glass_identification.txt"
    data=[]
    file=open(path)
    flag=0
    length_f=0
    for row in file:
        row=re.split(",|\n|\t",row)
        if(flag==0):
            length_f=(len(row)-1)
            flag=1
        if(row[length_f]==''):
            length_f-=1
        if(path!=path_exception1 and path!=path_exception2):
            row=row[0:length_f-1]
            data.append(row[0:(length_f)])
        else:
            data.append(row[1:(length_f+1)])

    data=np.asarray(data)
    return data.astype(np.float)

def get_tags(path):
    """En ésta función obtiene la última columna, donde en todos los archivos se sabe que están las etiquetas representativas de los grupos
    para traerlas todas y hacer una lista de ellas
    """
    print(path)
    data=[]
    file=open(path)
    flag=0
    length_f=0
    for row in file:
        row=re.split(",|\n|\t",row)
        if(flag==0):
            length_f=(len(row)-1)
            flag=1
        if(row[length_f]==''):
            row=row[0:length_f]
            length_f-=1
        if(path!="C:/Users/Luis/Desktop/Mineria 3/datasets/wine.txt"):
            data.append(row[length_f])
        else:
            data.append(row[0])
    data=np.asarray(data)
    return data.astype(int)

#Fin funciones k-means
def norm_Y(Y):
    y_set=list(set(Y))
    min_y=y_set[0]
    for i in range(0,len(y_set)):
        if(y_set[i]<min_y):
            min_y=y_set[i]
    if(min_y!=0):
        for i in range(0,len(Y)):
        	Y[i]-=1
    return Y

def S(i,clusters,centroids):
    sumatoria=0
    for element in clusters[i]:
        sumatoria+=np.sqrt(np.sum((element-centroids[i])**2,axis=None))
    promedio=sumatoria/len(clusters[i])
    return promedio

def d(centroid_i,centroid_j):
    distance=np.sqrt(np.sum((centroid_i-centroid_j)**2,axis=None))
    return distance

def db_index(centroids,clusters,n_population):
    maximos=np.zeros(len(clusters))
    for i in range(0,len(clusters)):
        m=[]
        for j in range(0,len(clusters)):
            if(i!=j):
                m.append((S(i,clusters,centroids)+S(j,clusters,centroids))/(d(centroids[i],centroids[j])))
        sorted(m)
        
        #print(m)
        maximos[i]=m[len(m)-1]
    maximos_sum=np.sum(maximos,axis=None)
    return maximos_sum

def euc_distance(v1,v2):
    v1,v2 = np.asarray(v1),np.asarray(v2)
    distance=np.sqrt(np.sum((v1-v2)**2,axis=None))
    return distance

def radios(centroids_list):
    centroids_list=np.asarray(centroids_list)

    mat_dis_centroids=[]
    for i in range(0,len(centroids_list)):
        mat_dis_centroids.append(np.zeros(len(centroids_list)))
    mat_dis_centroids=np.asarray(mat_dis_centroids)

    indxs=matriz_indices(len(centroids_list),len(centroids_list))
    
    for i in range(0,len(centroids_list)):
        for j in range(0,len(centroids_list)):
            if(i!=j):
                mat_dis_centroids[i][j]=euc_distance(centroids_list[i],centroids_list[j])

    sort_index(mat_dis_centroids,indxs)

    rads=np.zeros(len(centroids_list))

    for i in range(0,len(rads)):
        rads[i]=(mat_dis_centroids[i][1]+mat_dis_centroids[i][2])/2

    return rads
    """Se quejan porque Trump quiere construir un muro y sienten miedo cuando ven un hombre oscuro...
        Cuando me subo a un Uber preguntan de donde soy... Les digo "Venezuela" y de una nombran a Maduro...
    """

def RBF(X,c,radio):
    X=np.asarray(X)
    c=np.asarray(c)
    c=np.kron(np.ones((len(X),1)),c)
    retorno=np.exp(-np.sum((X-c)**2,axis=1)/(2*(radio)**2))
    return retorno

def Ytarget(Y,train_y):
	K=len(list(set(Y)))
	
	target=[]
	for i in range(0,len(train_y)):
		new_row=np.zeros(K)
		new_row[train_y[i]]=1
		target.append(new_row)
	target=np.asarray(target)

	return target

def M_1xM_2(m_1,m_2):
    m_return=[]
    for i in range(0,len(m_1)):
        fila=[]
        for k in range(0,len(m_2[0])):
            acum=0
            for j in range(0,len(m_1[i])):
                acum+=(m_1[i][j]*m_2[j][k])
            fila.append(acum)
        m_return.append(fila)
    m_return=np.asarray(m_return)

    return m_return

def soft_max(data):
    data=np.asarray(data)
    for i in range(0,len(data[0])):
        mean=stats.median(data[:,i])
        S=stats.pstdev(data[:,i])
        for j in range(0,len(data)):
            data[j][i]=(data[j][i]-mean)/S

    for i in range(0,len(data)):
        for j in range(0,len(data[i])):
            data[i,j]=(1-mt.exp(data[i,j]))/(1+mt.exp(data[i,j]))
    return data


def kmeans(path):
    folders=os.listdir(path)#Se crea una lista con los nombres de todas las carpetas dentro
    folders=np.asarray(folders)
    files=[]#inicializa lista para los nombres de los archivos pertenecientes a cada carpeta
    
    for f in folders:
        provisional_path=path+"/"+f#Se inicializa variable con la dirección anterior de la variable 'path' + el nombre de una carpeta 'x' dentro de StopSearch_2011_2017
        n_classes=get_n_classes(provisional_path)
        files.append([provisional_path,n_classes])

    files=np.asarray(files)
    count=0
    file_index=0
    
    #results=[]
    final_tags=[]

    for i in range(0,len(files)):
        file_results=[]
        prom_folds=[]
        for K in range(3,9):
            print("i: "+str(i)+", K: "+str(K))
            file_name=folders[i]
            X=np.asarray(get_data(files[i][0]))
            X=soft_max(X)
            #print(X)
            #time.sleep(5)
            file_tags=get_tags(files[i][0])

            X_shape=np.shape(X)
            k_index=np.random.choice(int(X_shape[0]),int(K),replace=False)#Selecciona indices aleatorios de elementos de la matriz original "X"
            centroids=X[k_index,:]#Asigna centroides aleatorios
            matriz_indx=matriz_indices(len(X),len(centroids))
            matriz_distancias=distances_matrix(X,centroids)
            sort_index(matriz_distancias,matriz_indx)

            
            clusters=iniciar_clusters(X,matriz_indx,K)
            old_centroids=centroids
            new_centroids=calculate_new_centroids(centroids,clusters)
            j=0
            while(olds_vs_news(old_centroids,new_centroids)):
                matriz_indx=matriz_indices(len(X),len(centroids))
                matriz_distancias=distances_matrix(X,centroids)
                sort_index(matriz_distancias,matriz_indx)
                clusters=iniciar_clusters(X,matriz_indx,K)
                old_centroids=centroids
                new_centroids=calculate_new_centroids(centroids,clusters)
                centroids=new_centroids
                j+=1
            #file_results.append(db_index(np.asarray(centroids),np.asarray(clusters),len(X)))
            #time.sleep(5)
            print("\n\n-------------------------------------\n\n")
            for element in matriz_indx:
                final_tags.append(element[0])

            data=X

            #time.sleep(5)

            k_folds=5
            all_index=np.arange(0,len(data))
            min_limit,max_limit=0,mt.ceil(len(data)/float(k_folds))
            indices_list=np.arange(min_limit,max_limit)
            radios_list=radios(centroids)
            Y=norm_Y(file_tags)
            const_limit=mt.ceil(len(data)/float(k_folds))
            cont_mcc=0
            for kf in range(0,k_folds):
                print("fold: "+str(kf+1))
                #print(str(min_limit)+","+str(max_limit))
                fold=np.arange(int(min_limit),int(max_limit))
                #print(fold)
                i_proof=data[fold,:]
                i_proof_y=Y[fold]
                i_proof=np.asarray(i_proof)
                i_proof_y=np.asarray(i_proof_y)
                train_index=np.setxor1d(fold,all_index)
                i_train=data[train_index,:]
                i_train_y=Y[train_index]

                phi_1=[]
                phi_1.append(np.reshape(np.kron(np.ones((len(i_train),1)),[1]),(len(i_train))))#ADD BIAS

                for cn in range(0,len(centroids)):
                    phi_1.append(RBF(i_train,centroids[cn],radios_list[cn]))
                phi_1=np.asarray(phi_1)
                phi_1=np.linalg.pinv(phi_1)
                phi_1=np.transpose(phi_1)
                #print(i_train_y)
                m_target=Ytarget(Y,i_train_y)
                """print(m_target)
                print("len: "+str(len(m_target)))
                time.sleep(5)"""
                W=M_1xM_2(phi_1,m_target)
                W=np.transpose(W)
                phi_2=[]
                phi_2.append(np.reshape(np.kron(np.ones((len(i_proof),1)),[1]),(len(i_proof))))
                for cn in range(0,len(centroids)):
                    phi_2.append(RBF(i_proof,centroids[cn],radios_list[cn]))
                phi_2=np.asarray(phi_2)
                phi_2=np.transpose(phi_2)
                
                y_net=[]
                #np_unhs=list(i_proof_y)
                len_set=len(list(set(Y)))
                #print(len_set)
                for d in range(0,len(phi_2)):
                    x=phi_2[d]
                    x=x.reshape(1,len(x))
                    x=np.transpose(x)
                    z=M_1xM_2(W,x)
                    x_tags=matriz_indices(1,len_set)
                    z=np.transpose(z)
                    sort_index(z,x_tags)
                    y_net.append(x_tags[0][len(x_tags)-1])
                y_net=np.asarray(y_net)
                cont_mcc+=matthews_corrcoef(i_proof_y,y_net)
                min_limit=max_limit
                max_limit+=const_limit
                if(max_limit>len(data)):
                    max_limit=len(data)-1
            prom_folds.append(cont_mcc/5)

        #GRAFICAR PROM_FOLDS a este nivel
        prom_folds=np.asarray(prom_folds)
        labels_k=["k-3","k-4","k-5","k-6","k-7","k-8"]
        #labels_x=[]
        index_mcc=np.arange(len(labels_k))
        #print("shape folds")
        #print(np.shape(prom_folds))
        #print(prom_folds)
        time.sleep(5)
        plt.subplots(figsize=(9,6))
        plt.xticks(index_mcc, labels_k,rotation="vertical")
        #plt.yticks(axis_y,labels_y)

        n=0
        plt.plot(prom_folds,color="r",marker="o", linestyle='--', label = "MCC")
        plt.title(folders[i])
        plt.savefig(folders[i]+".png")
        #plt.show()
        plt.close()


            #results.append(file_results)

        #clusters=np.asarray(clusters)
    #return results

def main():
    path="C:/Users/Luis/Desktop/Mineria 3/datasets"
    kmeans(path)#Esto es una lista de clusters separados para cada problema, ejemplo:
    


main()