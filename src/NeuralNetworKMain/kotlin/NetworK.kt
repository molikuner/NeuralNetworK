package com.molikuner.neuralnetwork

class NetworK private constructor(
    val inputLayer: InputLayer,
    val hiddenLayers: List<HiddenLayer>,
    val outputLayer: OutputLayer
) {
    init {
        run() // initializes weights randomly
    }
    private var running: Boolean = false
    fun run(vararg input: Activation) = run(input.toList())
    fun run(input: List<Activation>): List<CalculatedLayer> {
        // TODO make async calculating possible

        if (!running) {
            running = true
        } else throw ConcurrentModificationException("can't run the same neural network multiple times currently")
        try {
            if (input.isNotEmpty()) inputLayer.changeInput(input)
            val calculatingLayers: List<Layer<DefinedNeuron, *>> = hiddenLayers + outputLayer
            var prevLayer: CalculatedLayer = inputLayer.asCalculatedLayer
            return calculatingLayers.map { currentLayer ->
                with(prevLayer) {
                    currentLayer.calculate().also {
                        prevLayer = it
                    }
                }
            }
        } finally {
            running = false
        }
    }

    companion object {
        inline operator fun invoke(block: NetworKBuilder.() -> Unit): NetworK = NetworKBuilder().apply(block).build()
    }

    class NetworKBuilder {
        lateinit var inputLayer: InputLayer
        private val _hiddenLayers: MutableList<HiddenLayer> = mutableListOf()
        val hiddenLayers: List<HiddenLayer>
            get() = _hiddenLayers
        lateinit var outputLayer: OutputLayer

        inline fun inputLayer(
            block: InputLayer.InputLayerBuilder.() -> Unit = { numInputs = 2; layerBias = BiasNeuron() }
        ): InputLayer = InputLayer(block).also { inputLayer = it }
        fun hiddenLayer(
            block: HiddenLayer.HiddenLayerBuilder.() -> Unit = { neurons(2); layerBias = BiasNeuron() }
        ): HiddenLayer = HiddenLayer(block).also { _hiddenLayers.add(it) }
        fun hiddenLayers(
            num: Int,
            block: HiddenLayer.HiddenLayerBuilder.(index: Int) -> Unit = { neurons(num); layerBias = BiasNeuron() }
        ): List<HiddenLayer> = List(num) { HiddenLayer.HiddenLayerBuilder().apply { block(it) }.build() }.also { _hiddenLayers.addAll(it) }
        inline fun outputLayer(
            block: OutputLayer.OutputLayerBuilder.() -> Unit = { numOutputs = 1 }
        ): OutputLayer = OutputLayer(block).also { outputLayer = it }

        fun build(): NetworK = NetworK(inputLayer, hiddenLayers, outputLayer)
    }

    override fun toString(): String {
        return "${this::class.simpleName} { $inputLayer / ${hiddenLayers.size} { ${hiddenLayers.joinToString(" / ") { it.toString() }} } / $outputLayer }"
    }

    fun toLogString(): String = "${this::class.simpleName} { ${inputLayer.toLogString()} / ${hiddenLayers.size} { ${hiddenLayers.joinToString(" / ") { it.toLogString() }} } / ${outputLayer.toLogString()} }"
}
