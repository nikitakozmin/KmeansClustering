import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


NUM_POINTS = 50
NUM_RULES = 4


def cloud_generation_rule(start=0, end=1000):
    """Облако из рандомных значений"""
    return np.random.randint(start, end+1, 5)

def elongated_rounding_generation_rule():
    """Условный овал в 5-мерном пространстве"""
    u = np.random.uniform(0, 2*np.pi)
    v = np.random.uniform(0, 2*np.pi)
    x1 = 500*np.cos(u)
    x2 = 400*np.sin(u)*np.cos(v)
    x3 = 350*np.sin(u)*np.sin(v)
    x4 = 250*np.cos(u)*np.sin(v)
    x5 = 100*np.cos(v)
    return np.array([x1, x2, x3, x4, x5])

def line_generation_rule():
    """Линейная функция"""
    t = np.random.randint(0, 1000+1)
    x1 = 0 - t
    x2 = 0 + t
    x3 = 0 + t
    x4 = 0 + t
    x5 = 0 - t
    return np.array([x1, x2, x3, x4, x5])

def three_dimensional_visualization(points, num_points_for_rule, colors=["blue", "green", "red"]):
    """По умолчанию берёт три первых координаты и цвета по кругу"""
    num_rules = len(points)//num_points_for_rule
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i_rule in range(num_rules):
        ax.scatter(*zip(*map(lambda point: point[:3],
                             points[num_points_for_rule*i_rule:num_points_for_rule*(i_rule+1)])), 
                   c=colors[i_rule%len(colors)]
        )
    plt.show()

def two_dimensional_visualization(points, num_points_for_rule, colors=["blue", "green", "red"]):
    """По умолчанию берёт две первых координаты и цвета по кругу"""
    num_rules = len(points)//num_points_for_rule
    for i_rule in range(num_rules):
        plt.scatter(*zip(*map(lambda point: point[:2],
                              points[num_points_for_rule*i_rule:num_points_for_rule*(i_rule+1)])),
                    c=colors[i_rule%len(colors)]
        )
    plt.show()


if __name__ == '__main__':
    # Генерация точек по порядку для каждого правила
    print(f"Генерируем облака в размере {NUM_RULES} единиц...")
    points = np.zeros((NUM_POINTS*NUM_RULES, 5))
    for i in range(NUM_POINTS):
        points[i] = cloud_generation_rule()
        points[i+NUM_POINTS] = elongated_rounding_generation_rule()
        points[i+NUM_POINTS*2] = line_generation_rule()
        points[i+NUM_POINTS*3] = cloud_generation_rule(-1000, -250)

    # Отображение в 3D изначальных фигур, используя первых три координаты
    print("Генерируем изображение...")
    three_dimensional_visualization(points, NUM_POINTS)

    # Отображение в 2D изначальных фигур, используя первых две координаты
    print("Генерируем изображение...")
    two_dimensional_visualization(points, NUM_POINTS)

    # Использование PCA для точек
    print("Снижаем размерность...")
    pca = PCA(n_components=2)
    points_pca = pca.fit_transform(points)
    
    # Отображение в 2D фигур после PCA
    print("Генерируем изображение...")
    two_dimensional_visualization(points_pca, NUM_POINTS)
    
    print(f"Выполняем кластеризацию из {NUM_RULES} элементов...")
    kmeans = KMeans(n_clusters=NUM_RULES, random_state=0)
    kmeans.fit(points_pca)
    labels = kmeans.predict(points_pca)
    print("Генерируем изображение...")
    plt.scatter(*zip(*points_pca), c=labels, s=25)
    plt.scatter(*zip(*kmeans.cluster_centers_), marker='x', c="red", s=75)
    plt.show()
        
    # Проверка на "оптимальность" выбранного числа кластеров методом локтя
    wcss = []
    variations_num_clusters = list(range(1, 9))
    for num_clusters in variations_num_clusters:
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(points_pca)
        wcss.append(kmeans.inertia_)
    plt.plot(variations_num_clusters, wcss, marker='o')
    plt.show()
        

# # Тест PCA
# test = np.array(list(map(np.array,
#     [[135, 1500, 8.5, 200],
#     [165, 1600, 7.5, 220],
#     [150, 1550, 8.0, 210],
#     [120, 1450, 9.0, 190],
#     [170, 1650, 9.5, 220]
# ])))
# pca = PCA(n_components=2)
# test_pca = pca.fit_transform(test)
# plt.scatter(*zip(*test_pca), c="blue")
# plt.show()
