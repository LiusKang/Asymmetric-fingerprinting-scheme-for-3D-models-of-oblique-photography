import random
import math
from Crypto.Util.number import inverse


# 密钥生成

def KeyGen():
    # 随机选取两个独立的大素数 p 和 q
    p = generate_prime()
    q = generate_prime()

    # 计算 n = p * q
    n = p * q

    # 计算 𝜆 = lcm(p-1, q-1)
    lambd = lcm(p - 1, q - 1)

    # 选择 g = n + 1
    g = n + 1

    # 返回公钥 (n, g) 和私钥 (𝜆)
    pk = (n, g)
    sk = (lambd)

    return pk, sk


# 生成一个大素数

def generate_prime():
    while True:
        num = random.randint(2 ** 27, 2 ** 28)  # 生成一个随机数
        if is_prime(num):
            return num


# 检查一个数是否为素数

def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True


# 计算最小公倍数

def lcm(a, b):
    gcd = math.gcd(a, b)
    lcm = (a * b) // gcd
    return lcm


# 公钥 私钥

pk, sk = KeyGen()


# 加密

def encrypt(pk, plaintext):
    # 加密数字plaintext
    n, g = pk
    r = random.randint(1, n - 1)
    c = pow(g, plaintext, n ** 2) * pow(r, n, n ** 2) % (n ** 2)
    return c


# 解密

def decrypt(sk, pk, ciphertext):
    # 解密密文ciphertext
    n, g = pk
    lambda_n = sk
    x = pow(ciphertext, lambda_n, n ** 2)
    L = (x - 1) // n
    m = (L * inverse(lambda_n % n, n)) % n
    return m


if __name__ == '__main__':
    # 加密数字5和7
    c = encrypt(pk, 9999999999999)
    d = encrypt(pk, 0)

    # 同态加
    e = c * d

    # 解密
    dc = decrypt(sk, pk, c)
    dd = decrypt(sk, pk, d)
    de = decrypt(sk, pk, e)

    print("公钥 (pk):", pk)
    print("私钥 (sk):", sk)
    print(c)
    print(d)
    print(e)
    print(dc)
    print(dd)
    print(de)
