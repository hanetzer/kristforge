const char openclSource[] = R"ocl(
#ifdef VEC4
	typedef uint4 ui;
	typedef uchar4 uc;
	typedef long4 l;

	#define convert_ui(x) convert_uint4(x)
	#define convert_uc(x) convert_uchar4(x)
	#define convert_l(x) convert_long4(x)

	#define vload(x, y) vload4((x), (y))
	#define vstore(x, y, z) vstore4((x), (y), (z))
#elif defined(VEC2)
	typedef uint2 ui;
	typedef uchar2 uc;
	typedef long2 l;

	#define convert_ui(x) convert_uint2(x)
	#define convert_uc(x) convert_uchar2(x)
	#define convert_l(x) convert_long2(x)

	#define vload(x, y) vload2((x), (y))
	#define vstore(x, y, z) vstore2((x), (y), (z))
#else
	typedef uint ui;
	typedef uchar uc;
	typedef long l;

	#define convert_ui(x) convert_uint(x)
	#define convert_uc(x) convert_uchar(x)
	#define convert_l(x) convert_long(x)

	#define vload(x, y) (y)[(x)]
	#define vstore(x, y, z) (z)[(y)] = (x)
#endif

// right rotate macro
#ifdef BITALIGN
	#pragma OPENCL EXTENSION cl_amd_media_ops : enable
	#define RR(x, y) amd_bitalign((ui)x, (ui)x, (ui)y)
#else
	#define RR(x, y) rotate((ui)(x), -((ui)(y)))
#endif

// initial hash values
#define H0 0x6a09e667
#define H1 0xbb67ae85
#define H2 0x3c6ef372
#define H3 0xa54ff53a
#define H4 0x510e527f
#define H5 0x9b05688c
#define H6 0x1f83d9ab
#define H7 0x5be0cd19

// sha256 macros
#define CH(x,y,z) bitselect((z),(y),(x))
#define MAJ(x,y,z) bitselect((x),(y),(z)^(x))
#define EP0(x) (RR((x),2) ^ RR((x),13) ^ RR((x),22))
#define EP1(x) (RR((x),6) ^ RR((x),11) ^ RR((x),25))
#define SIG0(x) (RR((x),7) ^ RR((x),18) ^ ((x) >> 3))
#define SIG1(x) (RR((x),17) ^ RR((x),19) ^ ((x) >> 10))

__constant uint K[64] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2 };

void sha256_transform(uc *data, ui *H) {
	int i;
	ui a, b, c, d, e, f, g, h, t1, t2, m[64];

#pragma unroll
	for (i = 0; i < 16; i++) {
		m[i] = (convert_ui(data[i*4]) << 24) | (convert_ui(data[i*4+1]) << 16) | (convert_ui(data[i*4+2]) << 8) | convert_ui(data[i*4+3]);
		//printf("%v4i\n", m[i]);
		//printf("%i\n", m[i]);
	}

#pragma unroll
	for (i = 16; i < 64; i++)
		m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];
	a = H[0];
	b = H[1];
	c = H[2];
	d = H[3];
	e = H[4];
	f = H[5];
	g = H[6];
	h = H[7];

#pragma unroll
	for (i = 0; i < 64; i++) {
		t1 = h + EP1(e) + CH(e,f,g) + K[i] + m[i];
		t2 = EP0(a) + MAJ(a,b,c);
		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}

	H[0] += a;
	H[1] += b;
	H[2] += c;
	H[3] += d;
	H[4] += e;
	H[5] += f;
	H[6] += g;
	H[7] += h;
}

void sha256_finish(ui *H, uc *hash) {
	int i, l;

#pragma unroll
	for (i = 0; i < 4; i++) {
		l = 24 - i * 8;
		hash[i]      = convert_uc((H[0] >> l) & 0x000000ff);
		hash[i + 4]  = convert_uc((H[1] >> l) & 0x000000ff);
		hash[i + 8]  = convert_uc((H[2] >> l) & 0x000000ff);
		hash[i + 12] = convert_uc((H[3] >> l) & 0x000000ff);
		hash[i + 16] = convert_uc((H[4] >> l) & 0x000000ff);
		hash[i + 20] = convert_uc((H[5] >> l) & 0x000000ff);
		hash[i + 24] = convert_uc((H[6] >> l) & 0x000000ff);
		hash[i + 28] = convert_uc((H[7] >> l) & 0x000000ff);
	}
}

// sha256 digest of up to 55 bytes of input
// uchar data[64] - input bytes - will be modified
// uint inputLen - input length (in bytes)
// uchar hash[32] - output bytes - will be modified
void digest55(uc *data, uint inputLen, uc *hash) {
	// pad input
	data[inputLen] = 0x80;
	data[62] = (inputLen * 8) >> 8;
	data[63] = inputLen * 8;

	// init hash state
	ui H[8] = { H0, H1, H2, H3, H4, H5, H6, H7 };

	// transform
	sha256_transform(data, H);

	// finish
	sha256_finish(H, hash);
}

__kernel
void testDigest55(__global uchar *input, uint len, __global uchar *output) {
	uc in[64], out[32];

#pragma unroll
	for (int i = 0; i < 64; i++) in[i] = vload(i, input);

	digest55(in, len, out);

#pragma unroll
	for (int i = 0; i < 32; i++) vstore(out[i], i, output);
}

l score_hash(uc *hash) {
	return convert_l(hash[5]) + (convert_l(hash[4]) << 8) + (convert_l(hash[3]) << 16) + (convert_l(hash[2]) << 24) + (convert_l(hash[1]) << 32) + (convert_l(hash[0]) << 40);
}

long score_hash_scalar(uchar *hash) {
	return (hash[5]) + (hash[4] << 8) + (hash[3] << 16) + (((long)hash[2]) << 24) + (((long)hash[1]) << 32) + (((long)hash[0]) << 40);
}

__kernel
void testScore(__global uchar *hash, __global long *scores) {
	uc in[32];

#pragma unroll
	for (int i = 0; i < 32; i++) in[i] = vload(i, hash);

	l score = score_hash(in);

	vstore(score, 0, scores);
}

__constant l offset = (l)(0, 1, 2, 3);

__kernel
__attribute__((vec_type_hint(ui)))
void krist_miner(
	__global const uchar *kristAddress,  // 10 bytes
	__global const uchar *block,         // 12 bytes
	__global const uchar *prefix,        // 2 bytes
	const long offset,                   // convert to 10 bytes
	const long work,
	__global uchar *solution) {          // 12 bytes

	int id = get_global_id(0);
	long nonce = (id * VECSIZE) + offset;
	uc input[64];
	uc hashed[32];
	int i;

	// copy data to input

#pragma unroll
	for (i = 0; i < 10; i++) input[i] = kristAddress[i];

#pragma unroll
	for (i = 10; i < 22; i++) input[i] = block[i - 10];

#pragma unroll
	for (i = 22; i < 24; i++) input[i] = prefix[i - 22];

#pragma unroll
	for (i = 24; i < 34; i++) {

#if VECSIZE == 2 || VECSIZE == 4
		input[i] = convert_uc((l)(((nonce >> ((i - 24) * 5)) & 31) + 48) + offset);
#else
		input[i] = ((nonce >> ((i - 24) * 5)) & 31) + 48;
#endif
	}

	// hash it
	digest55(input, 34, hashed);

	// check for a solution

#if VECSIZE == 1
	if (score_hash(hashed) < work) {
		// copy data to output
#pragma unroll
		for (i = 0; i < 2; i++) solution[i] = prefix[i];

#pragma unroll
		for (i = 2; i < 12; i++) solution[i] = ((nonce >> ((i - 2) * 5)) & 31) + 48;
	}
#elif VECSIZE == 2
	// meh no one cares about vec2s
#elif VECSIZE == 4
	if (any(score_hash(hashed) < work)) {
#pragma unroll
		for (int n = 0; n < 4; n++) {
			uchar extracted[32];

			union {
				uchar component[VECSIZE];
				uc vector;
			} extractor;

#pragma unroll
			for (int i = 0; i < 32; i++) {
				extractor.vector = hashed[i];
				extracted[i] = extractor.component[n];
			}

			if (score_hash_scalar(extracted) < work) {
				// copy data to output
#pragma unroll
				for (i = 0; i < 2; i++) solution[i] = prefix[i];

#pragma unroll
				for (i = 2; i < 12; i++) {
					solution[i] = ((nonce >> ((i - 2) * 5)) & 31) + 48;
				}
			}
		}
	}
#else
#error Invalid vector size
#endif
}
)ocl";