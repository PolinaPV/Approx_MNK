from fastapi import FastAPI
from pydantic import BaseModel
import ApproxMNK

app = FastAPI()


class MNK(BaseModel):
    N: int
    sigma: float
    k: float
    b: float
    err: float


@app.put('/put_params')
def update_param(mnk: MNK):
    t = ApproxMNK.ApproxMNK(mnk.N, mnk.sigma, mnk.k, mnk.b)
    mnk.err = t.get_err() * 100
    return 'Средняя ошибка аппроксимации = {}%'.format(mnk.err)


@app.get('/')
def start():
    return 'API for Approx_MNK'

