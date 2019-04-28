package com.molikuner.neuralnetwork

sealed class Layer<out N : Neuron, out T : Neuron>(
    val neurons: List<N>,
    internal val typedNeurons: List<T>
) {
    operator fun plus(other: List<Layer<DefinedNeuron, DefinedNeuron>>): List<Layer<Neuron, Neuron>> = listOf(listOf(this), other).flatten()

    override fun toString(): String = "${this::class.simpleName} { ${neurons.joinToString(" / ")} }"

    open fun toLogString(): String = "${this::class.simpleName} { ${neurons.size} }"
}

sealed class BiasedLayer<T : DefinedNeuron>(neurons: List<T>, val layerBias: BiasNeuron?) :
        Layer<DefinedNeuron, T>(layerBias?.let { neurons + layerBias } ?: neurons, neurons) {
    override fun toLogString(): String = "${this::class.simpleName} { ${this.typedNeurons.size} + ${this.neurons.size - this.typedNeurons.size} }"
}

class InputLayer private constructor(inputs: List<InputNeuron>, layerBias: BiasNeuron?) : BiasedLayer<InputNeuron>(inputs, layerBias) {
    fun changeInput(inputs: List<Activation>) {
        if (typedNeurons.size != inputs.size) throw IllegalArgumentException("can't change the number of inputs currently")
        typedNeurons.forEachIndexed { index, inputNeuron ->
            inputNeuron._input = inputs[index]
        }
    }

    val asCalculatedLayer: CalculatedLayer
        get() = CalculatedLayer(neurons.map { (it as PredefinedNeuron).asActivatedNeuron })

    companion object {
        inline operator fun invoke(block: InputLayerBuilder.() -> Unit): InputLayer = InputLayerBuilder().apply(block).build()
    }

    class InputLayerBuilder {
        var layerBias: BiasNeuron? = null
        var forceBias: Boolean = true
        var numInputs: Int = 0
        fun build(): InputLayer = InputLayer(List(numInputs.also {
            if (it <= 0) throw IllegalArgumentException("can't create input layer without any inputs")
        }) { InputNeuron(0.0) }, layerBias.also {
            if (it == null && forceBias) throw IllegalArgumentException("layerBias missing, you need to specify explicitly if you want to allow that")
        })
    }
}

class HiddenLayer private constructor(neurons: List<HiddenNeuron>, layerBias: BiasNeuron?) : BiasedLayer<HiddenNeuron>(neurons, layerBias) {
    companion object {
        inline operator fun invoke(block: HiddenLayerBuilder.() -> Unit): HiddenLayer = HiddenLayerBuilder().apply(block).build()
    }

    class HiddenLayerBuilder {
        var layerBias: BiasNeuron? = null
        var forceBias: Boolean = true
        private val _neurons: MutableList<HiddenNeuron> = mutableListOf()
        val neurons: List<HiddenNeuron>
            get() = _neurons

        fun neuron(): HiddenNeuron = HiddenNeuron().also { _neurons.add(it) }

        fun neurons(num: Int): List<HiddenNeuron> = List(num) { HiddenNeuron() }.also { _neurons.addAll(it) }

        fun build(): HiddenLayer = HiddenLayer(neurons.also {
            if (it.isEmpty()) throw IllegalArgumentException("can't create layer without any neuron")
        }, layerBias.also {
            if (it == null && forceBias) throw IllegalArgumentException("layerBias missing, you need to specify explicitly if you want to allow that")
        })
    }
}

class OutputLayer private constructor(outputs: List<OutputNeuron>) : Layer<OutputNeuron, OutputNeuron>(outputs, outputs) {
    companion object {
        inline operator fun invoke(block: OutputLayerBuilder.() -> Unit): OutputLayer = OutputLayerBuilder().apply(block).build()
    }

    class OutputLayerBuilder {
        var numOutputs: Int = 0
        var neuronInitializer: (index: Int) -> OutputNeuron = { OutputNeuron() }
        fun build(): OutputLayer = OutputLayer(List(numOutputs.also {
            if (it <= 0) throw IllegalArgumentException("can't create output layer without any outputs")
        }, neuronInitializer))
    }
}

class CalculatedLayer(val states: List<ActivatedNeuron>) : Layer<ActivatedNeuron, ActivatedNeuron>(states, states) {
    fun Layer<DefinedNeuron, *>.calculate(): CalculatedLayer {
        // this@CalculatedLayer == prev layer
        // this@calculate == cur layer
        return CalculatedLayer(this@calculate.neurons.map { with(this@CalculatedLayer) { it.activation() } })
    }

    fun DefinedNeuron.activation(): ActivatedNeuron {
        // this@CalculatedLayer == prev layer
        // this@activation == cur neuron

        return ActivatedNeuron(when (this@activation) {
            is WeightableNeuron -> this@CalculatedLayer.neurons.sumByDouble { weightTo(it) * it.activation }.sigmoid
            is PredefinedNeuron -> this@activation.activation
        }, this@activation)
    }
}
