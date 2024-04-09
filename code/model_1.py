from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score


def train_and_evaluate_model_1(train_x, train_y, test_x, test_y, k=5):
    """
    KNN model
    """
    # train the model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_x, train_y)

    # evaluate on 'train.tagged'
    train_y_pred = knn.predict(train_x)
    train_f1 = f1_score(train_y, train_y_pred)
    print(f"\nF1 Score for Model 1 on 'train.tagged': {round(train_f1, 3)}\n")

    # evaluate on 'dev.tagged'
    test_y_pred = knn.predict(test_x)
    test_f1 = f1_score(test_y, test_y_pred)
    print(f"\nF1 Score for Model 1 on 'dev.tagged': {round(test_f1, 3)}\n")
