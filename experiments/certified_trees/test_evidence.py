"""Adversarial standard-library artifact checks. Synthetic parser fixtures only."""
import io,json,struct,tempfile,unittest,zipfile
from pathlib import Path
try:
    from . import audit as a
except ImportError:
    import audit as a


def fixture(shape=(2,3),order=False,descr='<i4',values=(1,2,3,4,5,6)):
    header=repr({'descr':descr,'fortran_order':order,'shape':shape}).encode('ascii')+b'\n'
    return b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header+struct.pack('<'+'i'*len(values),*values)

class EvidenceContracts(unittest.TestCase):
    def test_c_and_fortran_layout(self):
        c=a.npy(fixture())
        f=a.npy(fixture(order=True,values=(1,4,2,5,3,6)))
        self.assertEqual(c,f)
    def test_bad_json(self):
        for text in ('{"x":1,"x":2}','{"x":NaN}','{"x":Infinity}','{"x":1e999}'):
            with self.subTest(text=text),self.assertRaises(ValueError):a.loads(text)
    def test_bad_array_headers(self):
        for raw in (b'',b'\x93NUMPY',fixture()[:-1],fixture()+b'0',fixture(shape=(-2,3)),fixture(descr='|O8'),fixture(order=1),fixture(shape=(2,3,1),order=True)):
            with self.subTest(raw=raw[:30]),self.assertRaises(ValueError):a.npy(raw)
    def test_duplicate_npz(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'case.npz'
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with zipfile.ZipFile(p,'w') as z:z.writestr('a.npy',fixture());z.writestr('a.npy',fixture())
            with self.assertRaises(ValueError):a.arrays(p)
    def test_escape_and_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'good').write_text('a');(p/'link').symlink_to(p/'good')
            for name in ('../good','/etc/passwd','a\\b','link'):
                with self.subTest(name=name),self.assertRaises(ValueError):a.member(p,name)
            self.assertEqual(a.member(p,'good'),p/'good')
    def test_empty_incomplete_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'good').write_text('a')
            for record,required in (({},()),({'good':'0'*64},()),({'good':a.sha(p/'good')},('missing',))):
                with self.assertRaises(ValueError):a.identities(p,record,required)
    def test_derived_nonfinite_or_wrong_type(self):
        for v in (float('nan'),float('inf'),True,'1'):
            with self.assertRaises(ValueError):a.near(v,1.)
    def test_sha_bound_source_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'source.py').write_text('# first')
            record={'source.py':a.sha(p/'source.py')};a.identities(p,record,('source.py',))
            (p/'source.py').write_text('# second')
            with self.assertRaises(ValueError):a.identities(p,record,('source.py',))

if __name__=='__main__':unittest.main(verbosity=2)
