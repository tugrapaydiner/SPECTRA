"""Optional sklearn round-trip controls; no external datasets/downloads."""
import pytest
np = pytest.importorskip('numpy')
SVC = pytest.importorskip('sklearn.svm').SVC
from spectra.svm import Session
from spectra.svm_export import export_svc


@pytest.mark.parametrize('classes', [2, 3, 10])
def test_real_sklearn_roundtrip(tmp_path, library, classes):
    rng = np.random.default_rng(192 + classes)
    x = rng.normal(size=(classes * 16, 16)).astype('float32')
    model = SVC(C=10, gamma=0.1).fit(x, np.arange(len(x)) % classes)
    path = tmp_path / 'model.srt';export_svc(model, path)
    probe = np.concatenate((x, rng.normal(size=(64, 16)).astype('float32')))
    expected = model.predict(probe)
    for tables in (False, True):
        with Session(path, library, tables=tables) as session:
            assert [session.predict(row) for row in probe] == expected.tolist()
    with pytest.raises(FileExistsError):export_svc(model, path)


@pytest.mark.parametrize('kind', ['labels', 'width', 'kernel', 'ties', 'unfitted', 'type'])
def test_export_contract_rejection(tmp_path, kind):
    rng = np.random.default_rng(17);x = rng.normal(size=(32, 15 if kind == 'width' else 16))
    y = np.arange(len(x)) % 3
    if kind == 'labels':y += 10
    model = SVC(kernel='linear' if kind == 'kernel' else 'rbf', break_ties=kind == 'ties')
    if kind != 'unfitted':model.fit(x, y)
    with pytest.raises((ValueError, AttributeError)):
        export_svc(object() if kind == 'type' else model, tmp_path / 'bad.srt')
