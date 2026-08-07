import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D
from pathlib import Path
# need to use enviroment fmae_visual, tsne version too low

EV_DATASETS = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6']
BESS_DATASETS = ['b7']
LAB_DATASET = ['b10', 'b11', 'b12', 'b13']

ALL_DATASETS = EV_DATASETS + BESS_DATASETS + LAB_DATASET

DATASETS_MAPPING = {
    'b1': 'EV1', 'b2': 'EV2', 'b3': 'EV3', 'b4': 'EV4',
    'b5': 'EV5', 'b6': 'EV6', 'b7': 'BESS',
    'b10': 'MIT1', 'b11': 'THU', 'b12': 'MIT2', 'b13': 'KIT'
}

COLOR_MAP = {
    'greater than 95': {'color': '#94A3B8', 'val_min': 95, 'val_max': np.inf, 'pct': 0.25},
    '90-95':           {'color': '#416594', 'val_min': 90, 'val_max': 95,     'pct': 0.25},
    '85-90':           {'color': '#ECB426', 'val_min': 85, 'val_max': 90,     'pct': 0.25},
    'less than 85':    {'color': '#AC2124', 'val_min': 0,  'val_max': 85,     'pct': 0.25},
}

N_FOLDS = 5

TSNE_CONFIG = {
    'perplexity': 15,
    'n_iter': 2000,
    'early_exaggeration': 20,
    'learning_rate': 'auto',
    'init': 'pca',
    'n_jobs': 1,
    'random_state': 42,
}
script_dir = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = f'{Path(script_dir).parent}/data/capacity_embeddings'
OUTPUT_DIR = script_dir

MODELS = ['FMAE', 'LiPM']

NORMALIZER_MAP = {
    'b1':  46.32390322689564,
    'b2':  45.28971041043597,
    'b3':  44.034358340136066,
    'b4':  26.538070600693814,
    'b5':  35.473890427318786,
    'b6':  22.28958766646174,
    'b7':  95.9036963,
    'b10': 100,
    'b11': 100,
    'b12': 100,
    'b13': 100,
}


def load_raw_data(model_name, dataset_name):
    """
    load 5-fold raw_data
    return (features, targets, car_ids)
    """
    all_features = []
    all_targets = []
    all_car_ids = []

    for fold in range(N_FOLDS):
        feat_path = os.path.join(
            BASE_PATH, f'capacity_{model_name}_features_w_labels',
            f'features_{dataset_name}_f{fold}.npy'
        )
        targ_path = os.path.join(
            BASE_PATH, f'capacity_{model_name}_features_w_labels',
            f'targets_{dataset_name}_f{fold}.npy'
        )
        car_path = os.path.join(
            BASE_PATH, f'capacity_{model_name}_features_w_labels',
            f'car_id_{dataset_name}_f{fold}.npy'
        )

        features = np.load(feat_path)
        targets = np.load(targ_path)
        car_ids = np.load(car_path)

        all_features.append(features)
        all_targets.append(targets)
        all_car_ids.append(car_ids)

    features_all = np.vstack(all_features)
    
    # normlization
    norm_val = NORMALIZER_MAP.get(dataset_name, 100)
    targets_all = np.concatenate(all_targets) * 100 / norm_val * 100
    car_ids_all = np.concatenate(all_car_ids)

    return features_all, targets_all, car_ids_all


def get_qualified_cars(car_ids, targets, color_map, min_coverage=None):
    """
    return car ids that satisfies condition
    """
    if min_coverage is None:
        min_coverage = len(color_map) - 3

    unique_cars = np.unique(car_ids)
    qualified = []

    for car in unique_cars:
        car_mask = (car_ids == car)
        y_car = targets[car_mask]
        n_covered = 0
        for info in color_map.values():
            vmin, vmax = info['val_min'], info['val_max']
            if np.isinf(vmax):
                has_data = np.any(y_car >= vmin)
            else:
                has_data = np.any((y_car >= vmin) & (y_car < vmax))
            if has_data:
                n_covered += 1
        
        if n_covered >= min_coverage:
            qualified.append(car)

    return np.array(qualified)


def select_common_cars(car_ids_fmae, targets_fmae, car_ids_lipm, targets_lipm,
                       color_map, n_cars, rng):
    """
    select from the same cars
    return car_ids
    """
    qualified_fmae = get_qualified_cars(car_ids_fmae, targets_fmae, color_map)
    qualified_lipm = get_qualified_cars(car_ids_lipm, targets_lipm, color_map)

    common_cars = np.intersect1d(qualified_fmae, qualified_lipm)

    if len(common_cars) < n_cars:
        print(f"  Common qualified cars: {len(common_cars)} (need {n_cars}). SKIP.")
        return np.array([])

    return rng.choice(common_cars, n_cars, replace=False)


def filter_and_normalize(features, targets, car_ids, selected_cars):
    """
    filter and normalize
    """
    mask = np.isin(car_ids, selected_cars)
    if np.sum(mask) == 0:
        return None, None, None

    features_sel = features[mask]
    targets_sel = targets[mask]
    car_ids_sel = car_ids[mask]

    scaler = StandardScaler()
    features_sel = scaler.fit_transform(features_sel)

    return features_sel, targets_sel, car_ids_sel


def sample_by_SOH_per_car(features, targets, car_ids, num_per_car, color_map, rng):
    """
    sample data
    """
    unique_cars = np.unique(car_ids)
    if len(unique_cars) == 0:
        return (np.empty((0, features.shape[1]), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int32))

    X_out, y_out, c_out = [], [], []

    for car in unique_cars:
        car_mask = (car_ids == car)
        X_car = features[car_mask]
        y_car = targets[car_mask]

        for info in color_map.values():
            vmin, vmax = info['val_min'], info['val_max']
            if np.isinf(vmax):
                mask_bin = (y_car >= vmin)
            else:
                mask_bin = (y_car >= vmin) & (y_car < vmax)

            idx_bin = np.where(mask_bin)[0]
            if len(idx_bin) == 0:
                continue

            n_sample = min(max(1, int(num_per_car * info['pct'])), len(idx_bin))
            chosen = rng.choice(idx_bin, n_sample, replace=False)
            chosen = np.sort(chosen)

            X_out.append(X_car[chosen])
            y_out.append(y_car[chosen])
            c_out.append(np.full(len(chosen), car))

    if len(X_out) == 0:
        return (np.empty((0, features.shape[1]), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int32))

    return (np.vstack(X_out).astype(np.float32),
            np.concatenate(y_out).astype(np.float32),
            np.concatenate(c_out))


def run_tsne(features, cfg):
    """PCA + t-SNE, return 2D embeddings。"""
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(features)
    print(f"    PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    tsne = TSNE(
        n_components=2,
        random_state=cfg['random_state'],
        perplexity=cfg['perplexity'],
        n_iter=cfg['n_iter'],
        early_exaggeration=cfg['early_exaggeration'],
        learning_rate=cfg['learning_rate'],
        init=cfg['init'],
        n_jobs=cfg['n_jobs'],
    )
    embeddings = tsne.fit_transform(X_pca)
    print("    t-SNE done!")
    return embeddings


def plot_trajectory(embeddings, targets, brand_name, model_name, num_per_car, cfg, output_dir, car):
    """plot brand t-SNE graph"""
    fig, ax = plt.subplots(figsize=(8, 8))

    for range_str, info in COLOR_MAP.items():
        vmin, vmax = info['val_min'], info['val_max']
        if np.isinf(vmax):
            mask = (targets >= vmin)
        else:
            mask = (targets >= vmin) & (targets < vmax)

        if np.sum(mask) == 0:
            continue

        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=info['color'],
            marker='o',
            s=65,
            alpha=0.75,
            edgecolors='white',
            linewidths=0.3,
            zorder=2
        )

    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    filename = (
        f"{model_name}_{brand_name}_trajectory_"
        f"{car}car_{num_per_car}perCar_p{cfg['perplexity']}_i{cfg['n_iter']}_"
        f"e{cfg['early_exaggeration']}.png"
    )
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"    Saved: {output_path}")
    plt.show()
    plt.close(fig)


# ==================== 主流程 ====================

def main():
    num_per_car = 100   # samples per car
    num_cars = 15      # cars per brand

    rng = np.random.RandomState(42)

    for dataset_name in ALL_DATASETS:
        brand_name = DATASETS_MAPPING.get(dataset_name, dataset_name)
        print(f'\n========== Dataset: {brand_name} ({dataset_name}) ==========')

        # 1. Load raw data
        print("  Loading raw data...")
        features_fmae, targets_fmae, car_ids_fmae = load_raw_data('FMAE', dataset_name)
        features_lipm, targets_lipm, car_ids_lipm = load_raw_data('LiPM', dataset_name)

        # 2. find cars
        selected_cars = select_common_cars(
            car_ids_fmae, targets_fmae,
            car_ids_lipm, targets_lipm,
            COLOR_MAP, num_cars, rng
        )

        if len(selected_cars) == 0:
            print(f"  [SKIP] {dataset_name}: not enough common qualified cars.")
            continue

        print(f"  Selected {len(selected_cars)} common cars: {selected_cars}")

        # 3. tsne
        for model_name in MODELS:
            print(f"\n  --- Model: {model_name} ---")

            if model_name == 'FMAE':
                features, targets, car_ids = features_fmae, targets_fmae, car_ids_fmae
            else:
                features, targets, car_ids = features_lipm, targets_lipm, car_ids_lipm

            # filter and normlization
            features_sel, targets_sel, car_ids_sel = filter_and_normalize(
                features, targets, car_ids, selected_cars
            )

            if features_sel is None:
                print(f"    [SKIP] {model_name}: no data after filtering.")
                continue

            # sample
            rng_model = np.random.RandomState(42)  # sane random seeds for reprod
            X_s, y_s, c_s = sample_by_SOH_per_car(
                features_sel, targets_sel, car_ids_sel,
                num_per_car=num_per_car,
                color_map=COLOR_MAP,
                rng=rng_model,
            )

            if len(X_s) == 0:
                print(f"    [SKIP] {model_name}: no samples after SOH sampling.")
                continue

            print(f"    Sampled: {len(X_s)} points from {len(np.unique(c_s))} cars")

            # t-SNE
            embeddings = run_tsne(X_s, TSNE_CONFIG)

            # draw
            plot_trajectory(
                embeddings=embeddings,
                targets=y_s,
                brand_name=brand_name,
                model_name=model_name,
                num_per_car=num_per_car,
                car = num_cars,
                cfg=TSNE_CONFIG,
                output_dir=OUTPUT_DIR,
            )


if __name__ == '__main__':
    main()