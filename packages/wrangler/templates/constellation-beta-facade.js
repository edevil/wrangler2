// src/shim.ts
import worker, * as OTHER_EXPORTS from "__ENTRY_POINT__";
export * from "__ENTRY_POINT__";
import { Buffer } from "node:buffer";

const TENSOR_TYPES = ["float32", "int32", "float64", "int64", "string", "bool"];

function getTypeError(type) {
	return new Error(`unsupported type: ${type}`);
}

function is64BitType(type) {
	switch (type) {
		case "float64":
			return true;
		case "int64":
			return true;
		default:
			return false;
	}
}

function checkStringArray(stringArr) {
	const arr = new Array(stringArr.length);
	let s;
	for (let i = 0; i < stringArr.length; i++) {
		s = stringArr[i];
		if (typeof s !== "string" && !(s instanceof String)) {
			throw Error(`element ${s} is not a string`);
		}
		if (s instanceof String) {
			arr[i] = s.toString();
		} else {
			arr[i] = s;
		}
	}
	return arr;
}

function computeShapeNumEl(shape) {
	// shape should be array-like of integers
	var numel = 1;
	let d;
	for (let i = 0; i < shape.length; i++) {
		d = shape[i];
		if (typeof d !== "number" || !Number.isInteger(d)) {
			throw new Error(
				`expected shape to be array-like of integers but found non-integer element ${d}`
			);
		}
		numel *= d;
	}
	return numel;
}

function makeBigIntArray(intArr) {
	const bigIntArr = new BigInt64Array(intArr.length);
	for (let i = 0; i < intArr.length; i++) {
		bigIntArr[i] = BigInt(intArr[i]);
	}
	return bigIntArr;
}

function makeBooleanArray(boolArr) {
	const booleanArr = new Array(boolArr.length);
	for (let i = 0; i < boolArr.length; i++) {
		booleanArr[i] = Boolean(boolArr[i]);
	}
	return booleanArr;
}

function getConstructorForType(type) {
	switch (type) {
		case "float32":
			return (arr) => new Float32Array(arr);
		case "float64":
			return (arr) => new Float64Array(arr);
		case "int32":
			return (arr) => new Int32Array(arr);
		case "int64":
			return makeBigIntArray;
		case "string":
			return checkStringArray;
		case "bool":
			return makeBooleanArray;
		default:
			throw getTypeError(type);
	}
}

function int64Toint32Array(int64Array) {
	const arr = new Int32Array(int64Array.length);
	let n;
	for (let i = 0; i < int64Array.length; i++) {
		n = int64Array[i];
		if (n < Number.MIN_SAFE_INTEGER || n > Number.MAX_SAFE_INTEGER) {
			throw new Error(`element ${n} is too big to represent as int32`);
		}
		arr[i] = Number(n);
	}
	return arr;
}

function float64toFloat32Array(float64Array) {
	const arr = new Float32Array(float64Array.length);
	for (let i = 0; i < float64Array.length; i++) {
		arr[i] = Math.fround(float64Array[i]);
	}
	return arr;
}

function b64ToArray(base64, type) {
	const buffer = Buffer.from(base64, "base64");

	const arrBuffer = new ArrayBuffer(buffer.length);
	const fullView = new Uint8Array(arrBuffer);
	for (let i = 0; i < buffer.length; i++) {
		fullView[i] = buffer[i];
	}
	switch (type) {
		case "float32":
			// if (buffer.length % 4 != 0) {
			//   throw Error(`Invalid number of bytes (${buffer.length}) for input of type ${type}`)
			// }
			return new Float32Array(arrBuffer);
		case "float64":
			return new Float64Array(arrBuffer);
		case "int32":
			return new Int32Array(arrBuffer);
		case "int64":
			return new BigInt64Array(arrBuffer);
		default:
			throw Error(`invalid data type for base64 input: ${type}`);
	}
}

function arrayToB64(arr) {
	return new Buffer.from(arr.buffer).toString("base64");
}

function isConsnTensor(t) {
	let tensorProps = ["value", "type", "shape"];
	for (let i = 0; i < tensorProps.length; i++) {
		if (typeof t[tensorProps[i]] === "undefined") {
			return false;
		}
	}
	return true;
}

export class Tensor {
	constructor(type, shape, value, name) {
		// throws type error on invalid types
		var typedArr = getConstructorForType(type)(value);

		// check shape validity
		let numel;
		try {
			numel = computeShapeNumEl(shape);
		} catch (error) {
			throw new Error(`invalid shape: ${error}`);
		}
		if (numel != typedArr.length) {
			throw new Error(
				`invalid shape: expected ${numel} elements for shape ${shape} but value array has length ${typedArr.length}`
			);
		}

		this.name = name;
		this.type = type;
		this.value = typedArr;
		this.shape = shape;
		this.b64Value = null;
		this.is64Bit = is64BitType(type);
	}

	static fromBase64Value(type, shape, b64Value, name) {
		const value = b64ToArray(b64Value, type);
		const tensor = new Tensor(type, shape, value, name);
		tensor.b64Value = b64Value;
		return tensor;
	}

	static fromJSON(obj) {
		var { type, shape, value, b64Value, name } = obj;
		if (value) {
			return new Tensor(type, shape, value, name);
		} else {
			return Tensor.fromBase64Value(type, shape, b64Value, name);
		}
	}

	static fromORT(tensor) {
		var { type, dims, data } = tensor;
		return new Tensor(type, dims, data);
	}

	as32Bit() {
		switch (this.type) {
			case "float64":
				return new Tensor(
					"float32",
					this.shape,
					float64toFloat32Array(this.value)
				);
			case "int64":
				return new Tensor("int32", this.shape, int64Toint32Array(this.value));
			default:
				return this;
		}
	}

	toJSON(encode64 = false) {
		// if datatype is int64/float64 reduce to int32/float32 or base64 encode
		if (encode64 && this.is64Bit) {
			if (this.b64Value === null) {
				this.b64Value = arrayToB64(this.value);
			}
			return {
				type: this.type,
				shape: this.shape,
				value: null,
				b64Value: this.b64Value,
				name: this.name,
			};
		} else {
			var as32 = this.as32Bit();
			return {
				type: as32.type,
				shape: this.shape,
				value: Array.from(as32.value),
				b64Value: null,
				name: this.name,
			};
		}
	}
}

// src/index.ts
var ConstellationApi = class {
	constructor(binding) {
		this.binding = binding;
	}
	async query(modelId, inputs) {
		if (isConsnTensor(inputs)) {
			inputs = [inputs];
		} else if (!Array.isArray(inputs)) {
			const inputArr = new Array();
			// assume it is a record of Tensors by input name
			const inputNames = Object.keys(inputs);
			let t;
			for (let i = 0; i < inputNames.length; i++) {
				// set name for each tensor
				t = inputs[inputNames[i]];
				if (!isConsnTensor(t)) {
					throw Error(`Found non-tensor type in input map: ${typeof t}`);
				}
				t.name = inputNames[i];
				inputArr.push(t);
			}
			inputs = inputArr;
		}
		var inputJSON = new Array(inputs.length);
		for (let i = 0; i < inputs.length; i++) {
			inputJSON[i] = inputs[i].toJSON(true);
		}
		const jsonBody = { model: modelId, input: inputJSON };
		const body = JSON.stringify(jsonBody);

		const res = await this.binding.fetch("/run", {
			method: "POST",
			body: body,
		});
		if (!res.ok) {
			throw new Error(`API returned ${res.status}: ${await res.text()}`);
		}

		const output = await res.json();
		const decodedOut = {};
		const outputKeys = Object.keys(output);
		for (let i = 0; i < outputKeys.length; i++) {
			decodedOut[outputKeys[i]] = Tensor.fromJSON(output[outputKeys[i]]);
		}

		return decodedOut;
	}
};

// src/shim.ts
var CONSTELLATION_IMPORTS = __CONSTELLATION_IMPORTS__;
var CONSTELLATION_BETA_PREFIX = `__CONSTELLATION_BETA__`;
var envMap = /* @__PURE__ */ new Map();
function getMaskedEnv(env) {
	if (envMap.has(env)) return envMap.get(env);
	const newEnv = new Map(Object.entries(env));
	CONSTELLATION_IMPORTS.filter((bindingName) =>
		bindingName.startsWith(CONSTELLATION_BETA_PREFIX)
	).forEach((bindingName) => {
		newEnv.delete(bindingName);
		const newName = bindingName.slice(CONSTELLATION_BETA_PREFIX.length);
		const newBinding = new ConstellationApi(env[bindingName]);
		newEnv.set(newName, newBinding);
	});
	const newEnvObj = Object.fromEntries(newEnv.entries());
	envMap.set(env, newEnvObj);
	return newEnvObj;
}
var shim_default = {
	...worker,
	async fetch(request, env, ctx) {
		return worker.fetch(request, getMaskedEnv(env), ctx);
	},
	async queue(batch, env, ctx) {
		return worker.queue(batch, getMaskedEnv(env), ctx);
	},
	async scheduled(controller, env, ctx) {
		return worker.scheduled(controller, getMaskedEnv(env), ctx);
	},
	async trace(traces, env, ctx) {
		return worker.trace(traces, getMaskedEnv(env), ctx);
	},
	async email(message, env, ctx) {
		return worker.email(message, getMaskedEnv(env), ctx);
	},
};
export { shim_default as default };
