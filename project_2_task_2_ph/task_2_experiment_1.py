from experiment_methods import prepare_data_experiment_one, evaluate_model
from models import MNISTCNN2, Cifar10CNN2, DatasetName, MNISTCNN, Cifar10CNN

if __name__ == '__main__':

    # For models with two features
    train_loader1, test_loader1, model_path1 = prepare_data_experiment_one(DatasetName.MNIST, two_features=True)
    train_loader2, test_loader2, model_path2 = prepare_data_experiment_one(DatasetName.CIFAR10, two_features=True)

    mnist_model_2 = MNISTCNN2()
    cifar10_model_2 = Cifar10CNN2()

    evaluate_model(mnist_model_2, train_loader1, test_loader1, model_path1, DatasetName.MNIST, "Two Features", with_feature_scatter=True, visualised_sample_size=100)
    evaluate_model(cifar10_model_2, train_loader2, test_loader2, model_path2, DatasetName.CIFAR10, "Two Features", with_feature_scatter=True, visualised_sample_size=100)

    # For full models
    train_loader3, test_loader3, model_path3 = prepare_data_experiment_one(DatasetName.MNIST, two_features=False)
    train_loader4, test_loader4, model_path4 = prepare_data_experiment_one(DatasetName.CIFAR10, two_features=False)

    mnist_model = MNISTCNN()
    cifar10_model = Cifar10CNN()

    evaluate_model(mnist_model, train_loader3, test_loader3, model_path3, DatasetName.MNIST, "", visualised_sample_size=None)
    evaluate_model(cifar10_model, train_loader4, test_loader4, model_path4, DatasetName.CIFAR10, "", visualised_sample_size=None)