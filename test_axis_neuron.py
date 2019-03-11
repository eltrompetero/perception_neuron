from .axis_neuron import *
import pandas as pd

def test_extract_AN_port():
    """Test coordinate system transformation."""
    # Create motion dataframe.
    headers = ['Timestamp']+calc_file_headers()[:-1]
    X = np.ones((10,len(headers)))
    df = pd.DataFrame(X,columns=headers)

    t,x,v,a = extract_AN_port(df,'left')
    assert x[0][0,0]==1
    assert x[0][0,1]==-1
    assert x[0][0,2]==-1

    t,x,v,a = extract_AN_port(df,'right')
    assert x[0][0,0]==1
    assert x[0][0,1]==-1
    assert x[0][0,2]==-1

    t,x,v,a = extract_AN_port(df,'left',rotation_angle=np.pi/2)
    assert np.isclose(x[0][0,0],1)
    assert np.isclose(x[0][0,1],1)
    assert x[0][0,2]==-1

if __name__=='__main__':
    test_extract_AN_port()
