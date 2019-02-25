from dlcliche.image import *
from dlcliche.math import *
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import axes3d
from IPython.display import display

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import albumentations as A

sys.path.append('..') # app
sys.path.append('../..') # root
from few_shot.extmodel_proto_net_clf import ExtModelProtoNetClf
from app_utils_clf import *


def get_test_images(data_test):
    return sorted([str(f).replace(data_test+'/', '') for f in Path(data_test).glob('*.jpg')])


def get_aug(re_size=224, to_size=224, train=True):
    augs = [A.Resize(height=re_size, width=re_size)]
    if train:
        augs.extend([
            A.RandomCrop(height=to_size, width=to_size),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=30, p=0.75),
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.75),
            A.Blur(p=0.5),
            A.Cutout(max_h_size=to_size//12, max_w_size=to_size//12, p=0.5),
        ])
    else:
        augs.extend([A.CenterCrop(height=to_size, width=to_size)])
    return A.Compose(augs + [A.Normalize()])


def get_img_loader(folder, to_gray=False):
    def _loader(filename):
        img = load_rgb_image(folder + '/' + str(filename))
        if to_gray:
            img = np.mean(img, axis=-1).astype(np.uint8)
            img = np.stack((img,)*3, axis=-1)
        return img
    return _loader


class WhaleImages(Dataset):
    def __init__(self, path, images, labels, re_size=256, to_size=224, train=True):
        self.datasetid_to_filepath = images
        self.datasetid_to_class_id = labels
        self.classes = sorted(list(set(labels)))
        
        self.df = pd.DataFrame({'class_id':labels, 'id':list(range(len(images)))})

        self.loader = get_img_loader(path, to_gray=True)
        self.transform = get_aug(re_size=re_size, to_size=to_size, train=train)
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, item):
        instance = self.loader(self.datasetid_to_filepath[item])
        instance = self.transform(image=instance)['image']
        instance = self.to_tensor(instance)
        label = self.datasetid_to_class_id[item]
        return instance, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.cls2imgs)


def plot_prototype_2d_space_distribution(prototypes):
    X = prototypes
    pca = PCA(n_components=2)
    X_pca = pca.fit(X).transform(X)
    print('PCA: Explained variance ratio: %s'
          % str(pca.explained_variance_ratio_))
    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=.6)
    plt.title('Prototype Distribution PCA')
    plt.xlim((-4, 4))
    plt.ylim((-3, 3))
    plt.show()
    return X_pca


def plot_prototype_3d_space_distribution(prototypes):
    X = prototypes
    pca = PCA(n_components=3)
    X_pca = pca.fit(X).transform(X)
    print('PCA: Explained variance ratio: %s'
          % str(pca.explained_variance_ratio_))
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(X_pca[:, 0],X_pca[:, 1],X_pca[:, 2])
    ax.set_title('Prototype Distribution PCA')
    ax.set_xlim((-4, 4))
    ax.set_ylim((-3, 3))
    ax.set_zlim((-3, 3))
    plt.show()
    return X_pca


def get_classes(data='data', except_new_whale=True, append_new_whale_last=True):
    df = pd.read_csv(data+'/train.csv')
    if except_new_whale:
        df = df[df.Id != 'new_whale']
    classes = sorted(list(set(df.Id.values)))
    if append_new_whale_last:
        classes.append('new_whale')
    return classes


def calculate_results(weight, SZ, get_model_fn, device, train_csv='data/data.csv',
                      data_train='data/train', data_test='data/test'):
    # Training samples
    df = pd.read_csv(train_csv)
    df = df[df.Id != 'new_whale']
    images = df.Image.values
    labels = df.Id.values

    # Test samples
    test_images = get_test_images(data_test)
    dummy_test_gts = list(range(len(test_images)))

    print(f'Training samples: {len(images)}, # of labels: {len(list(set(labels)))}.')
    print(f'Test samples: {len(test_images)}.')
    print(f'Work in progress for {weight}...')

    def get_dl(images, labels, folder, SZ=SZ, batch_size=64):
        ds = WhaleImages(folder, images, labels, re_size=SZ, to_size=SZ, train=False)
        dl = DataLoader(ds, batch_size=batch_size)
        return dl

    # Make prototypes
    trn_dl = get_dl(images, labels, data_train)
    model = get_model_fn(device=device, weight_file=weight+'.pth')
    proto_net = ExtModelProtoNetClf(model, trn_dl.dataset.classes, device)

    proto_net.make_prototypes(trn_dl)

    # Calculate distances
    test_dl = get_dl(test_images, dummy_test_gts, data_test)
    test_embs, gts = proto_net.get_embeddings(test_dl)
    test_dists = proto_net.predict_embeddings(test_embs, softmax=False)

    np.save(f'test_dists_{weight}.npy', test_dists)
    np.save(f'prototypes_{weight}.npy', np.array([x.mean() for x in proto_net.prototypes]))


# Thanks to https://github.com/radekosmulski/whale/blob/master/utils.py
def top_5_preds(preds): return np.argsort(preds.numpy())[:, ::-1][:, :5]

def top_5_pred_labels(preds, classes):
    top_5 = top_5_preds(preds)
    labels = []
    for i in range(top_5.shape[0]):
        labels.append(' '.join([classes[idx] for idx in top_5[i]]))
    return labels


def prepare_submission(submission_filename, test_dists, new_whale_thresh, data_test, classes):
    def _create_proto_submission(preds, name, classes):
        sub = pd.DataFrame({'Image': get_test_images(data_test)})
        sub['Id'] = [classes[i] if not isinstance(i, str) else i for i in 
                     top_5_pred_labels(torch.tensor(preds), classes)]
        ensure_folder('subs')
        sub.to_csv(f'subs/{name}.csv.gz', index=False, compression='gzip')

    dist_new_whale = np.ones_like(test_dists[:, :1])
    dist_new_whale[:] = new_whale_thresh
    final_answer = np.c_[test_dists, dist_new_whale]

    _create_proto_submission(final_answer, submission_filename, classes)
    print(submission_filename,
          pd.read_csv(f'subs/{submission_filename}.csv.gz').Id.str.split().apply(lambda x: x[0] == 'new_whale').mean(),
          len(set(pd.read_csv(f'subs/{submission_filename}.csv.gz').Id.str.split().apply(lambda x: x[0]).values)))
    display(pd.read_csv(f'subs/{submission_filename}.csv.gz').head())
