"""
This is the similar implementation of agent with reAct prompt but how under the hood tool calling is handled
as shown previously in ice-breaker

Manual Step-by-Step Execution:

Uses a while loop to manually handle each step of the agent's thought process
Explicitly tracks intermediate steps in memory
Manages the back-and-forth between thinking and tool execution


Text-Based Parsing:

Uses a specific text format (Question/Thought/Action/Action Input/Observation)
Relies on ReActSingleInputOutputParser to extract actions from the text output
The agent's thinking is explicitly included in the prompt via "agent_scratchpad"


Tool Management:

Tools are manually executed using a helper function (find_tool_by_name)
Results are formatted and added to intermediate steps


"""


from typing import Union, List

from dotenv import load_dotenv
from langchain.agents.format_scratchpad import format_log_to_str
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool, tool
from langchain.tools.render import render_text_description
from langchain.agents.output_parsers import ReActSingleInputOutputParser

from langchain_ollama import ChatOllama

from callbacks import AgentCallbackHandler

load_dotenv()


# @tool will convert this textlength method to a langchain tool
# which can be used by agents.
# the multiline comment description of the tool is important, thats what the llm refers to to
# understand what the tool does.
@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip(
        '"'
    )  # stripping away non-alphabetic characters just in case

    return len(text)


def find_tool_by_name(tools: List[BaseTool], tool_name: str) -> BaseTool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")


if __name__ == "__main__":
    print("Hello ReAct LangChain!")
    tools = [get_text_length]

    # this template is by harrison chase
    # this prompt triggers the chain of thought process. (reAct paper )
    # this prompt let the llm know about the tools available and select among those
    # to get to the desired output.
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """
    # this {agent_scratchpad} is to store the previous iteration of the llm.

    """
    with partial we are inputting the static variables in the prompt
    meaning this variable won't change in every iteration.
    """
    prompt = PromptTemplate.from_template(template=template).partial(
        # render_text_description is used to simply convert the tool names to text, as the llm only accepts text as input.
        tools=render_text_description(tools),
        # joining all the tool names with , as a seperator.
        tool_names=", ".join([t.name for t in tools]),
    )

    # llm = ChatOpenAI(temperature=0, stop=["\nObservation", "Observation"])
    llm = ChatOllama(
        model="mistral",
        temperature=0,

        # this is the hault point in the llm response.
        # if it finds text=Observation, it won't generate that token anymore
        # we need this because:
        # if it generate more token then it would generate the Final answer token text too
        # along with the action and action input token text.
        # which will confuse the ReActSingleInputOutputParser()
        # as it can only accept either one of those type of token as input not both.
        stop=["\nObservation", "Observation"],

        # this will log all the calls to and responses from the llm.
        callbacks=[AgentCallbackHandler()],
    )

    # to store the previous llm iteration data and results.
    intermediate_steps = []

    # 'input' is the dynamic variable in the prompt template which is the actual question
    # from the user. we have to provide it in a dict format.
    # IMPORTANTLY: the question prompt can change how the llm answers, it follows the
    # format mentioned in prompt template, if it doesn't, it won't create the
    # \nAction:
    # \nActionInput:
    # type format
    # which ReActSingleInputOutputParser() expects. so it will throw an error.

    """
    these are dynamic variables in the prompt which are needed to be passed like this in a 
    dict. so that in every iteration it can change. when we invoke the agent.
    """

    """ also could have done :
    
    agent = (prompt | llm | ReActSingleInputOutputParser())

    .... 
    agent_step = agent.invoke({
        "input": "What is the length of 'DOG' in characters?",
        "agent_scratchpad": format_log_to_str(intermediate_steps),
    })

    same thing. 

    it takes the input variables and puts in the prompt in its places.
    """
    agent = (
            {
                "input": lambda x: x["input"],
                # we have to use langchain's format_log_to_str() method, because llm, don't understand
                # the AgentAction type that the tuple contains in the list.
                # so this converts those llm unreadable types to strings. by properly traversing
                # and converting.
                "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
            }
            | prompt
            | llm
            | ReActSingleInputOutputParser()
        # and ReActSingleInputOutputParser() returns a AgentAction object, if it receives a format:
        # Action:
        # ActionInput:
        # else if it receives:
        # Final Answer:
        # then it returns a AgentFinish type.
        # can't receive both the types though.
        # (if it does, then edit llm prompt so that it doesn't return something like that)
        # which holds the tool, tool_name etc.
        # in a format:
        # tool=''
        # tool_input=''
        # log=''
        # etc.
    )

    # The Union type in Python is used for type hinting and specifies that a variable can be of multiple types.
    # similar to Either in dart.
    # also the dict key  = the input variable used in the prompt template ( 'input' in this case )
    # IMPORTANTLY: the question prompt can change how the llm answers, does it follow the
    # format mentioned in prompt template, if it doesn't it won't create the action and action
    # input formatted output, which ReActSingleInputOutputParser() expects. so it will throw an error.

    # agent_step = agent.invoke(
    #     {
    #         "input": "What is the length of 'DOG' in characters?",
    #     }
    # )
    # print(agent_step)

    agent_step = ""
    # iterate as many times as the llm is not including the final answer
    # in its response. hence, not returning a AgentFinish.
    while not isinstance(agent_step, AgentFinish):
        agent_step = agent.invoke(
            {
                # if this input also included a input variable we would have to use another PromptTemplate
                # with simple input variable
                """ e.g:  prompt_template.format_prompt(name_of_person=name) """
                "input": "What is the length of 'DOG' in characters?",
                "agent_scratchpad": intermediate_steps,
            }
        )

        print(agent_step)

        # isInstance checks if agent_step is an instance of the class AgentAction
        if isinstance(agent_step, AgentAction):
            # this is the tool execution function.
            # llm just got the prompt which includes tools and tool names and the input question.
            # then it thought and decided which tool to use
            # returns that in a particular format that ReActSingleInputOutputParser() expects.
            # and that returns a AgentAction type.
            # then here we are executing the tool.
            tool_name = agent_step.tool  # the tool name that agent selected.
            tool_to_use = find_tool_by_name(tools, tool_name)  # finding that tool
            tool_input = agent_step.tool_input  # getting the question input

            observation = tool_to_use.func(str(tool_input))  # executing the tool.
            print(f"{observation=}")  # answer.

            # storing the previous data, llm thoughts and the tool execution result.
            # sending a tuple to the append method.
            # so this becomes a List[Tuple(AgentAction, str)]
            intermediate_steps.append((agent_step, str(observation)))

    # agent 2nd iteration.
    # as in the second iteration we are providing the previous iteration's response and stuff
    # it will not repeat already performed steps. like using tools, & observation: .
    # so it will this time generate a final answer.
    # hence ReActSingleInputOutputParser will give a AgentFinish response.

    # USING THE WHILE LOOP NOW, SO DON'T NEED TO MANUALLY CALL THE 2ND ITERATION.
    # agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
    #     {
    #         "input": "What is the length of the word: DOG",
    #         "agent_scratchpad": intermediate_steps,
    #     }
    # )

    if isinstance(agent_step, AgentFinish):
        print("### AgentFinish ###")
        print(agent_step.return_values["output"])
