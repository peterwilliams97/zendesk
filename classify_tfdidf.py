import ast
from collections import defaultdict
import random
import sys
import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
from config import CUSTOM_FIELDS_KEY, RANDOM_SEED
from config_keywords import PRODUCT_KINDS
from utils import load_text, truncate, round_score, SummaryReader, directory_ticket_numbers

PRODUCT_NAMES = sorted(PRODUCT_KINDS.keys(), reverse=True)
PRODUCT_INDEXES = {k: i for i, k in enumerate(PRODUCT_NAMES)}

def describe(X):
    "Returns a string describing the shape, dtype, and type of the input array `X`."
    try:
        return f"{list(X.shape)}:{X.dtype}"
    except:
        return f"@@type={type(X)}"

AVOID = ["Inbound call from", "Conversation with", "Chat with", "Chat started", "Chat ended"]

def avoidable(text):
    "Ensure that the text is not a conversation or chat."
    return any(text.startswith(s) for s in AVOID)

def load_comments(paths, subject, max_comments=2, max_size=100_000_000):
    """
    Load comments from multiple paths and concatenate them into a single text.

    Args:
        paths (list): A list of file paths to load comments from.
        subject (str): The subject of the comments.
        max_size (int, optional): The maximum size of the concatenated text. Defaults to 100_000_000.

    Returns:
        str: The preprocessed text containing the concatenated comments.
    """
    texts = [f"{subject}::-::"]
    size = 0
    for path in paths:
        text = load_text(path)
        if avoidable(text):
            continue
        texts.append(text)
        size += len(text)
        if size > max_size:
            assert False, f"size={size} > max_size={max_size}"
            break
        if len(texts) >= max_comments + 1:
            break
    return "\n\n".join(texts)

def data_to_Xy(data_list, get_features):
    """
    Convert a list of data into feature matrix X and target vector y.

    Args:
        data_list (list): A list of data, where each element is a tuple containing metadata and comments.

    Returns:
        X (numpy.ndarray): Feature matrix of shape (n_samples,), where each element is a loaded comment.
        y (numpy.ndarray): Target vector of shape (n_samples,), where each element is a label.

    """
    count_vect = CountVectorizer(stop_words="english")

    metadata_list, comments_list = zip(*data_list)

    print(f"PRODUCT_INDEXES={len(PRODUCT_INDEXES)}")
    for i, kind in enumerate(PRODUCT_KINDS.keys()):
        print(f"{i:2}: {kind}")

    index_label = []
    for i, meta in enumerate(metadata_list):
        subject = meta["subject"]
        if avoidable(subject):
            continue
        fields = meta[CUSTOM_FIELDS_KEY]
        fields = ast.literal_eval(fields)
        product_name = fields.get(25019086)
        if not product_name:
            continue
        product_kind = None
        for kind, words in PRODUCT_KINDS.items():
            for word in words:
                if word in product_name.lower():
                    product_kind = kind
                    break
            if product_kind:
                break
        if not product_kind or product_kind not in PRODUCT_INDEXES:
            continue
        label = PRODUCT_INDEXES[product_kind]
        index_label.append((i, label, subject))

    print(f"data_list={len(data_list)}")
    print(f"index_label={len(index_label)} {index_label[:10]}")

    X = np.empty(len(index_label), dtype=object)
    y = np.empty(len(index_label), dtype=int)
    for i, (idx, label, subject) in enumerate(index_label):
        X[i] = get_features(comments_list[idx])
        y[i] = label

    print(f"X={describe(X)}")
    print(f"y={describe(y)}")

    assert len(X) and len(y), f"len(X)={len(X)} != len(y)={len(y)}"

    return X, y

#
# Try out a few sklearn classifiers on the ticket data.
#
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import LinearSVC

VECTORISER_PARAMS = {'ngram_range': (1, 2), 'min_df': 5, 'max_df': 0.8}

CLASSIFIERS = {
    "SGDClassifier": (SGDClassifier, {'alpha': 1e-05, 'penalty': 'l2', 'loss': 'log_loss'}),
    # "GradientBoostingClassifier": (GradientBoostingClassifier, {}),
    # "RandomForestClassifier": (RandomForestClassifier, {}),
    "RidgeClassifier": (RidgeClassifier, {"alpha": 1.0, "solver": "sparse_cg"}),
    "LogisticRegression": (LogisticRegression, {"C": 5, "max_iter": 1000}),
    # "ComplementNB": (ComplementNB, {"alpha": 0.1}),
    # "KNeighborsClassifier": (KNeighborsClassifier, {"n_neighbors": 100}),
    # "NearestCentroid": (NearestCentroid, {}),
    # "LinearSVC": (LinearSVC, {"C": 0.1, "dual": False, "max_iter": 1000}),
}

def make_vectoriser():
    "Create and return a pipeline for vectorizing and transforming text data as TF-IDF."
    return Pipeline([
        ("vect",  CountVectorizer(**VECTORISER_PARAMS)),
        ("tfidf", TfidfTransformer()),
    ])

def make_pipeline(random_state, cls_name):
    """
    Create a pipeline for text classification.

    Parameters:
    - random_state (int): Random seed for reproducibility.
    - cls_name (str): Name of the classifier to use.

    Returns:
    - pipeline (Pipeline): Text classification pipeline.

    """
    clf_type, params = CLASSIFIERS[cls_name]
    try:
        clf = clf_type(**params, random_state=random_state)
    except TypeError:
        clf = clf_type(**params)
    return Pipeline([
        ("vect",  CountVectorizer(**VECTORISER_PARAMS)),
        ("tfidf", TfidfTransformer()),
        ("clf",   clf),
    ])

def informative_words(pipeline, X_train_in, n):
    """
    Extracts informative words from the training data using a pipeline.

    Args:
        pipeline: The pipeline object containing the vectorizer and classifier.
        X_train_in: The input training data.
        n: The number of informative words to extract per product.

    Returns:
        df: A pandas DataFrame containing the informative words and their scores.

    Raises:
        AssertionError: If the number of features does not match the number of products.

    """
    import pandas as pd

    vectoriser = make_vectoriser()
    X_train = vectoriser.fit_transform(X_train_in)
    words = vectoriser.named_steps["vect"].get_feature_names_out()

    clf = pipeline.named_steps["clf"]
    average_feature_effects = clf.coef_ * np.asarray(X_train.mean(axis=0)).ravel()

    assert average_feature_effects.shape[0] == len(PRODUCT_INDEXES), (
            f"features={average_feature_effects.shape}" +
            f" != products={len(PRODUCT_INDEXES)}.\n" +
            "    Use more training data or fewer products.")

    pd_dict = {}
    predictive_words = {}
    for name in PRODUCT_NAMES:
        k = PRODUCT_INDEXES[name]
        results = [(word, average_feature_effects[k, i]) for i, word in enumerate(words)]
        results.sort(key=lambda x: -x[1])
        for word, score in results[:n]:
            predictive_words[word] = score

        score_name = f"{k}_score"
        words, scores = zip(*results)
        pd_dict[name.upper()] = words
        pd_dict[score_name] = scores

    df = pd.DataFrame(pd_dict)
    print(df)
    return df

def train_evaluate(pipeline, X_train, y_train, X_test, y_test, N=20, M=150):
    """
    Train a TF-IDF classification pipeline on the training set, evaluate its the  on the test set
    and print the metrics.

    Args:
        pipeline (sklearn.pipeline.Pipeline): The pipeline to evaluate.
        X_train (array-like): The training data.
        y_train (array-like): The target labels for the training data.
        X_test (array-like): The test data.
        y_test (array-like): The target labels for the test data.
        N (int, optional): The number of top words to display. Defaults to 20.
        M (int, optional): The maximum number of words to consider. Defaults to 150.

    Returns:
        tuple: A tuple containing the micro-averaged F1 score, the time taken for fitting the pipeline,
               a set of wrong comments, and the top informative words.
    """
    clf = pipeline.named_steps["clf"]

    t0 = time.time()
    pipeline.fit(X_train, y_train)
    dt = time.time() - t0
    y_pred = pipeline.predict(X_test)
    micro_f1 = f1_score(y_test, y_pred, average="micro")
    confusion = confusion_matrix(y_test, y_pred)

    print(f"Micro-averaged F1 score on test set: {micro_f1:2f} ({dt:.1f} seconds)")
    print(confusion)
    print("-" * 40)

    wrong = y_pred != y_test
    wrong_comments = set(X_test[wrong])

    top_words = informative_words(pipeline, X_train, n=20)

    return micro_f1, dt, wrong_comments, top_words

def run_train_eval(pipeline, X, y):
    """
    Trains and evaluates a classification pipeline on the given dataset.

    Args:
        pipeline (sklearn.pipeline.Pipeline): The pipeline to train and evaluate.
        X (array-like): The input features.
        y (array-like): The target labels.

    Returns:
        dict: A dictionary containing the evaluation results.
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y,)
    except ValueError:
        print(f"Not enough data for training. X={list(X.shape)} y={list(y.shape)}", file=sys.stderr)
        raise

    clf = pipeline.named_steps["clf"]
    print(f"Training and evaluating {clf.__class__.__name__} on {len(X_train)} samples.")
    return train_evaluate(pipeline, X_train, y_train, X_test, y_test)

def classify_tickets_all_classifiers(data_list, seed):
    """
    Classifies tickets using multiple classifiers.

    Args:
        data_list (list): A list of data to be classified.
        seed (int): Random seed for reproducibility. Defaults to RANDOM_SEED.
    """
    X, y = data_to_Xy(data_list)

    print("=" * 80)

    assert data_list, "No data to classify."

    results = {}
    for classifier in sorted(CLASSIFIERS.keys()):
        random.seed(seed)
        np.random.seed(seed)
        random_state = check_random_state(seed)
        pipeline = make_pipeline(random_state, classifier)
        results[classifier] = run_train_eval(pipeline, X, y)

    classifiers = sorted(results.keys(), key=lambda k: (
                    -round_score(results[k][0], 2),
                    round_score(results[k][1], 0),
                    -round_score(results[k][0], 3),
                    k.lower()))
    print("=" * 80)
    for classifier in classifiers:
        micro_ft, dt, _, _ = results[classifier]
        print(f"{classifier:30}: {micro_ft:.3f} {dt:5.1f} seconds")

    wrongest = defaultdict(int)
    for classifier in classifiers[:4]:
        _, _, wrong, _ = results[classifier]
        for comment in wrong:
            wrongest[comment] += 1
    print("=" * 80)
    print(f"Most frequently misclassified comments: {len(wrongest)} total examples")
    for comment, count in sorted(wrongest.items(), key=lambda x: -x[1])[:60]:
        print(f"{count:3}: {repr(truncate(comment, 150))}")

def classify_tickets(zd, summariser, ticket_numbers, classifier="LogisticRegression", seed=RANDOM_SEED):
    """
    Classify tickets using a specified classifier.

    Parameters:
    - data_list (list): A list of data to be classified.
    - classifier (str): The classifier to be used. Default is "LogisticRegression".
    - seed (int): The random seed for reproducibility. Default is RANDOM_SEED.

    If the classifier is set to "all", the function will use all available classifiers
    to classify the tickets.

    The function first converts the data into feature matrix X and target vector y.
    It then initializes the random seed and random state for reproducibility.
    Finally, it creates a pipeline with the specified classifier and runs the
    training and evaluation process.
    """
    SECTION_NAMES = ["SUMMARY",
                     "PRODUCT", "FEATURES", "CLASS", "DEFECT",
                     "DESCRIPTION", "CHARACTERISTICS", "PROBLEMS"]

    reader = SummaryReader(SECTION_NAMES)
    ticket_numbers = directory_ticket_numbers(summariser.summary_dir)
    # assert False, f"ticket_numbers={len(ticket_numbers)} {type(ticket_numbers[0])} {ticket_numbers[:5]}"

    def get_feature(ticket_number):
        path = summariser.summary_path(ticket_number)
        text = load_text(path)
        sections = reader.summary_to_sections(text)
        get = lambda key: sections.get(key, "not specified")
        rows = []
        for name in SECTION_NAMES:
            sep = "\n" if name == "PROBLEMS" else " "
            val = get(name)
            rows.append(f"{name}: {val}")
        return "\n\n".join(rows)

    # data_list = [(zd.metadata(t), zd.comment_paths(t)) for t in ticket_numbers]
    data_list = [(zd.metadata(t), t) for t in ticket_numbers]
    data_list = [both for both in data_list if both[1]]
    assert ticket_numbers, f"No comments to classify in {len(ticket_numbers)} tickets."

    if classifier == "all":
        return classify_tickets_all_classifiers(data_list, seed=seed)

    X, y = data_to_Xy(data_list, get_feature)

    random.seed(seed)
    np.random.seed(seed)
    random_state = check_random_state(seed)

    pipeline = make_pipeline(random_state, classifier)
    run_train_eval(pipeline, X, y)
