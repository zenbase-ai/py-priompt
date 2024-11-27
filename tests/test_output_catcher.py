from priompt.output_catcher import OutputCatcher


def test_add_output_no_priority():
    output_catcher = OutputCatcher[str]()
    output = "Test Output"

    output_catcher.on_output(output)

    outputs = output_catcher.get_outputs()
    assert output in outputs


def test_add_output_with_priority():
    output_catcher = OutputCatcher[str]()
    output = "Test Output"
    priority = 5

    output_catcher.on_output(output, {"p": priority})

    outputs = output_catcher.get_outputs()
    assert output in outputs
    assert output_catcher.get_output() == output


def test_outputs_correct_order():
    output_catcher = OutputCatcher[str]()

    output_catcher.on_output("output1", {"p": 2})
    output_catcher.on_output("output2")
    output_catcher.on_output("output3", {"p": 3})
    output_catcher.on_output("output4")

    outputs = output_catcher.get_outputs()
    assert outputs == ["output3", "output1", "output2", "output4"]


def test_get_first_output():
    output_catcher = OutputCatcher[str]()
    output_catcher.on_output("Test1", {"p": 1})
    output_catcher.on_output("Test2", {"p": 2})
    output_catcher.on_output("Test3", {"p": 3})

    first_output = output_catcher.get_output()
    outputs = output_catcher.get_outputs()
    assert first_output == outputs[0]


def test_get_output_empty():
    output_catcher = OutputCatcher[str]()
    output = output_catcher.get_output()
    assert output is None


def test_multiple_outputs_same_priority():
    output_catcher = OutputCatcher[str]()

    output_catcher.on_output("output1", {"p": 1})
    output_catcher.on_output("output2", {"p": 1})
    output_catcher.on_output("output3", {"p": 1})

    outputs = output_catcher.get_outputs()
    assert outputs == ["output1", "output2", "output3"]


def test_multiple_outputs_different_priorities():
    output_catcher = OutputCatcher[str]()

    output_catcher.on_output("output1", {"p": 2})
    output_catcher.on_output("output2", {"p": 1})
    output_catcher.on_output("output3", {"p": 3})

    outputs = output_catcher.get_outputs()
    assert outputs == ["output3", "output1", "output2"]


def test_multiple_outputs_no_priority():
    output_catcher = OutputCatcher[str]()

    output_catcher.on_output("output1")
    output_catcher.on_output("output2")
    output_catcher.on_output("output3")

    outputs = output_catcher.get_outputs()
    assert outputs == ["output1", "output2", "output3"]


def test_empty_outputs():
    output_catcher = OutputCatcher[str]()
    outputs = output_catcher.get_outputs()
    assert outputs == []


def test_single_output():
    output_catcher = OutputCatcher[str]()
    output = "Test Output"
    output_catcher.on_output(output)

    outputs = output_catcher.get_outputs()
    assert len(outputs) == 1
    assert outputs[0] == output


def test_mixed_priority_and_no_priority():
    output_catcher = OutputCatcher[str]()

    output_catcher.on_output("output1")
    output_catcher.on_output("output2")
    output_catcher.on_output("output3")
    output_catcher.on_output("output4", {"p": 1})
    output_catcher.on_output("output5", {"p": 2})

    outputs = output_catcher.get_outputs()
    assert outputs[:2] == ["output5", "output4"]
    assert outputs[2:] == ["output1", "output2", "output3"]
