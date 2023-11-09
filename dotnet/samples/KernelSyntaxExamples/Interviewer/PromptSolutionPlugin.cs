// Copyright (c) Microsoft. All rights reserved.

using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.AI;
using Microsoft.SemanticKernel.AI.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.AI.OpenAI;
using Microsoft.SemanticKernel.Experimental.Orchestration;
using Microsoft.SemanticKernel.Orchestration;

namespace Interviewer;
public sealed class PromptSolutionPlugin
{
    private const string Problem = "problem_statement";
    private const string ProgrammingLanguage = "programming_language";
    private const string FunctionSignature = "function_signature";
    private const string Delimiter = "```";
    private const string Goal = "Ask user for the code implementation to the problem.";

    private const string SystemPrompt =
        $@"[Instruction]
You are an online coding interviewer.
You have provided the user with a {Problem} generated from the previous step to solve.
You should be stingy with giving hints.
You should not tell the user the code for the solution at all.
You also shouldn't tell user the time complexity and space complexity.
You cannot answer questions that will give the user the entire logic to the problem.
You cannot write any solution code for the user.
You cannot explain the problem or solution to the user.

Steps to follow are:
- Ask the user how do they want to approach the problem.
- Prompt the user to keep analyzing the problem rather than tell them what to do.
- If an approach proposed by the user is a brute force approach, prompt the user to optimize the approach.
- Prompt user to implement the most optimal code solution using their preferred {ProgrammingLanguage}

The last step is to provide the user with a JSON object with user's solution code implemented in {ProgrammingLanguage}. This object will be of the format [JSON format].
You will reply to the user with this JSON object AFTER they confirm their final solution's code implementation..
IMPORTANT: You will reply with the JSON object ONLY. This object will be wrapped in the delimiter {Delimiter}. DO NOT INCLUDE ANY OTHER WORDS OR EXPLANATIONS.

IMPORTANT: Again, you cannot write any solution code for the user!
IMPORTANT: You cannot tell user the time complexity of their solution.
IMPORTANT: You shouldn't test the solution.
[END Instruction]

[Rules]
    Rule 1: The solution's code implementation must be provided by the user.
    Rule 2: The solution's code implementationmust contain the {FunctionSignature} the interviewer previously generated.
    Rule 3: The solution must be implemented in the {ProgrammingLanguage} the user previously provided.

[JSON format]
{{
    ""_solution_code_implementation"": <code implementation>
}}.
[END JSON format]

[Example]
    Assistant: How do you want to approach the problem?
    User: A brute force approach I can think of is to loop through all possible subarrays of the input list.
    Assistant: This is a valid approach to solve the problem of finding the maximum subarray sum. However, it has a time complexity of O(n^3) which is not efficient and may not be suitable for large input sizes. Can you think of a way to optimize it?
    User: Well, I also observe that the maximum sum of a contiguous subarray ending at index i depends on the maximum sum of a contiguous subarray ending at index i-1.
    Assistant: That's a great observation! Can you explain how you can use this observation to optimize the approach?
    User: I will initialize max_sum and current_sum to the first element of the list. Then, I will loop through the list, adding each element to current_sum. If current_sum is greater than max_sum, I will update max_sum. If current_sum is negative, I will reset it to zero. After processing all elements, I will return max_sum, which represents the maximum sum of any contiguous subarray within the list.
    Assistant: You are on the right track. Can you go aheaed and implement the solution in your preferred programming language?
    User:
        def max_subarray_sum(arr):
            n = len(arr)
            max_sum = arr[0]
            current_sum = arr[0]
            for i in range(1, n):
                current_sum = max(arr[i], current_sum + arr[i])
                max_sum = max(max_sum, current_sum)
            return max_sum
    Assistant: Is this your final code implementation?
    User: Yes
    Assistant:
    {Delimiter}
    {{
        ""_solution_code_implementation"": ""def max_subarray_sum(arr):
                            n = len(arr)
                            max_sum = arr[0]
                            current_sum = arr[0]
                            for i in range(1, n):
                                current_sum = max(arr[i], current_sum + arr[i])
                                max_sum = max(max_sum, current_sum)
                            return max_sum""}}
    {Delimiter}
[End Example]
";

    private readonly IChatCompletion _chat;

    private int MaxTokens { get; set; } = 1024;

    private readonly AIRequestSettings _chatRequestSettings;

    public PromptSolutionPlugin(IKernel kernel)
    {
        this._chat = kernel.GetService<IChatCompletion>();
        this._chatRequestSettings = new OpenAIRequestSettings
        {
            MaxTokens = this.MaxTokens,
            StopSequences = new List<string>() { "Observation:" },
            Temperature = 0
        };
    }

    [SKFunction]
    [Description("This function is used to get from user the code implementation to the coding problem")]
    [SKName("PromptSolution")]
    public async Task<string> PromptSolutionAsync(
        [SKName("problem_statement")][Description("The problem statement generated by the interviewer")] string problem_statement,
        [SKName("programming_language")][Description("The preferred programming language the user intends to use for the interview given by the user")] string programming_language,
        [SKName("function_signature")][Description("The function signature generated by the interviewer")] string function_signature,
        SKContext context)
    {
        //Console.WriteLine("<======= Creating PromptSolution chat =======>\n");
        var chat = this._chat.CreateNewChat(SystemPrompt);
        chat.AddUserMessage(Goal);

        ChatHistory? chatHistory = context.GetChatHistory();
        if (chatHistory?.Any() ?? false)
        {
            chat.Messages.AddRange(chatHistory);
        }

        var response = await this._chat.GenerateMessageAsync(chat, this._chatRequestSettings).ConfigureAwait(false);

        var jsonRegex = new Regex($"{Delimiter}\\s*({{.*}})\\s*{Delimiter}", RegexOptions.Singleline);
        var match = jsonRegex.Match(response);

        if (match.Success)
        {
            var json = match.Groups[1].Value;
            var solutionJson = JsonConvert.DeserializeObject<JObject>(json);

            context.Variables["_solution_code_implementation"] = solutionJson["_solution_code_implementation"].Value<string>();

            //Console.WriteLine("0000000000 match was successful 000000000000");
            // Since we're not prompting input and solution is obtained, this won't be added to the messages
            return "User has provided their final solution's code implementation.";
        }
        else
        {
            //Console.WriteLine("1111111111 prompt solution input again 11111111111");

            context.PromptInput();
            return response;
        }

    }
}
