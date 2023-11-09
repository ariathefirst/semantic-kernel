// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using Newtonsoft.Json.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Azure.Search.Documents.Models;
using Google.Apis.CustomSearchAPI.v1.Data;
using Kusto.Cloud.Platform.Utils;
using Kusto.Data.Common;
using Microsoft.Extensions.Logging;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.AI;
using Microsoft.SemanticKernel.AI.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.AI.OpenAI;
using Microsoft.SemanticKernel.Experimental.Orchestration;
using Microsoft.SemanticKernel.Orchestration;
using Microsoft.SemanticKernel.Plugins.Core;
using Microsoft.SemanticKernel.Plugins.Memory;
using Microsoft.SemanticKernel.Plugins.Web;
using Microsoft.SemanticKernel.Plugins.Web.Bing;
using Microsoft.SemanticKernel.TemplateEngine.Basic;
using NRedisStack.Graph.DataTypes;
using StackExchange.Redis;
using System.Drawing;

/**
 * This example shows how to use FlowOrchestrator to execute a given flow with interaction with client.
 */

// ReSharper disable once InconsistentNaming
public static class Example64_FlowOrchestrator
{
    private static readonly Flow s_flow = FlowSerializer.DeserializeFromYaml(@"
    name: FlowOrchestrator_Example_Flow
    goal: generate interview problem and conduct coding interview with user
    steps:
      - goal: Generate the coding problem prompt that outputs the maximum subarray in a list.
        plugins:
          - GenerateProblemPlugin
        provides:
          - problem_statement
      - goal: Ask user what programming language they will use to implement the solution. Generate the function signature they should implement the solution in.
        plugins:
          - CollectPreferredLanguageToolPlugin
          - GenerateFunctionSignaturePlugin
        requires:
          - problem_statement
        provides:
          - programming_language
          - function_signature
      - goal: Ask user to code the problem solution. Get fully implemented solution from user without validating.
        plugins:
          - PromptSolutionPlugin
        requires:
          - problem_statement
          - programming_language
          - function_signature
        provides:
          - _solution_code_implementation
      - goal: Prompt user to analyze time and space complexity of their final solution. 
        plugins:
          - PromptTimeAndSpaceComplexityAnalysisPlugin
        requires:
          - problem_statement
          - _solution_code_implementation
        provides:
          - time_complexity
          - space_complexity
      - goal: Give user the decision of the interview based on user's interview performance.
        plugins:
          - ConcludeInterviewPlugin
        requires:
          - problem_statement
          - _solution_code_implementation
          - time_complexity
          - space_complexity
        provides:
          - interview_analysis
          - decision
");
    public static Task RunAsync()
    {
        return RunExampleAsync();
    }

    private static async Task RunExampleAsync()
    {
        var bingConnector = new BingConnector(TestConfiguration.Bing.ApiKey);
        var webSearchEnginePlugin = new WebSearchEnginePlugin(bingConnector);
        using var loggerFactory = LoggerFactory.Create(loggerBuilder =>
            loggerBuilder
                .AddConsole()
                .AddFilter(null, LogLevel.Error));

        Dictionary<object, string?> plugins = new()
        {
            { webSearchEnginePlugin, "WebSearch" },
        };
        FlowOrchestrator orchestrator = new(
            GetKernelBuilder(loggerFactory),
            await FlowStatusProvider.ConnectAsync(new VolatileMemoryStore()).ConfigureAwait(false),
            plugins,
            config: GetOrchestratorConfig());

        var sessionId = Guid.NewGuid().ToString();

        Console.WriteLine("*****************************************************");
        Stopwatch sw = new();
        sw.Start();
        Console.WriteLine("Flow: " + s_flow.Name);
        var result = await orchestrator.ExecuteFlowAsync(s_flow, sessionId, "Execute the flow").ConfigureAwait(false);

        Console.WriteLine("Assistant: " + result.ToString());

        string[] userInputs = new[]
        {
            "python",
            "Are there any constraints on the size of the input and can I assume the input will always be valid?",
            "Do I need to implement in real code or just pseudo code is enough?",
            "I can loop through all possible subarrays of the input list but that might not be optimal.",
            "Well, I also observe that the maximum sum of a contiguous subarray ending at index i depends on the maximum sum of a contiguous subarray ending at index i-1.",
            "Can I start with the pseudo code?",
            "I will initialize max_sum and current_sum to the first element of the list. Then, I will loop through the list, adding each element to current_sum. If current_sum is greater than max_sum, I will update max_sum. If current_sum is negative, I will reset it to zero. After processing all elements, I will return max_sum, which represents the maximum sum of any contiguous subarray within the list.",
            "My solution is: \ndef max_subarray_sum(arr):\n n = len(arr)\n max_sum = arr[0]\n current_sum = arr[0]\n    for i in range(1, n):\n        current_sum = max(arr[i], current_sum + arr[i])\n        max_sum = max(max_sum, current_sum)\n    return max_sum",
            "Yes.",
            "I think the time complexity is O(n) since I iterate through the array exactly once and the space complexity is O(1) since there's no extra space used.",
            "Yes",
            "So how did I do on the interview?",
        };

        foreach (var t in userInputs)
        {
            Console.WriteLine($"\nUser: {t}");
            result = await orchestrator.ExecuteFlowAsync(s_flow, sessionId, t).ConfigureAwait(false);
            Console.WriteLine("Assistant: " + result.ToString());
        }

        Console.WriteLine("Time Taken: " + sw.Elapsed);
        Console.WriteLine("*****************************************************");
    }
    private static FlowOrchestratorConfig GetOrchestratorConfig()
    {
        var config = new FlowOrchestratorConfig
        {
            MaxStepIterations = 40
        };

        return config;
    }

    private static KernelBuilder GetKernelBuilder(ILoggerFactory loggerFactory)
    {
        var builder = new KernelBuilder();

        return builder
            .WithAzureOpenAIChatCompletionService(
                TestConfiguration.AzureOpenAI.ChatDeploymentName,
                TestConfiguration.AzureOpenAI.Endpoint,
                TestConfiguration.AzureOpenAI.ApiKey,
                true,
                setAsDefault: true)
            .WithRetryBasic(new()
            {
                MaxRetryCount = 3,
                UseExponentialBackoff = true,
                MinRetryDelay = TimeSpan.FromSeconds(3),
            })
            .WithLoggerFactory(loggerFactory);
            //.WithLoggerFactory(
            //    LoggerFactory.Create(option => option.AddConsole()));
    }

    public sealed class GenerateProblemPlugin
    {
        private const string Goal = "Generate coding problem prompt of finding maximum subarray in a list";
        private const string SystemPrompt =
            "I am a question generating bot. I will describe the coding problem " +
            "of finding maximum subarray in a list for the user to solve.";

        private readonly IChatCompletion _chat;

        private int MaxTokens { get; set; } = 256;

        private readonly AIRequestSettings _chatRequestSettings;

        public GenerateProblemPlugin(IKernel kernel)
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
        [Description("This function is used to generate a coding problem")]
        [SKName("GenerateProblem")]
        public async Task<string> GenerateProblemAsync(
            [SKName("problem_statement")][Description("The coding problem prompt")] string problem,
            SKContext context)
        {
            //Console.WriteLine("<======= Creating GenerateProblem chat =======>\n");
            var chat = this._chat.CreateNewChat(SystemPrompt);
            chat.AddUserMessage(Goal);

            ChatHistory? chatHistory = context.GetChatHistory();
            if (chatHistory?.Any() ?? false)
            {
                chat.Messages.AddRange(chatHistory);
            }

            if (!string.IsNullOrEmpty(problem))
            {
                context.Variables["problem_statement"] = problem;

                Console.WriteLine("Assistant: Hello! Thanks for joining the coding interview. " +
                   "Here's the problem for you to solve: \n" + problem);

                return "Assistant: Hello! Thanks for joining the coding interview. " +
                   "Here's the problem for you to solve: \n" + problem;
            }

            return "Assistant: Hello! Thanks for joining the coding interview. " +
                   "Here's the problem for you to solve: \n" +
                   await this._chat.GenerateMessageAsync(chat, this._chatRequestSettings).ConfigureAwait(false);
        }
    }

    public sealed class CollectPreferredLanguageToolPlugin
    {
        private const string Goal = "Get programming language from user.";

        private const string SystemPrompt =
            "Your only task is to get the name of the programming language the user intends to use. " +
            "You can only ask user what programming language they will use to solve the given problem. " +
            "You cannot respond to input from the user about solving the problem. " +
            "You cannot solve the problem for the user or provide any hints or snippets of code. " +
            "If the user responds with anything other than a programming language, say that you don't know. " +
            "You cannot answer questions that will give the user the entire logic to the problem. " +
            "You cannot write any solution code for the user. " +
            "You cannot explain the problem or solution to the user.";
        private readonly IChatCompletion _chat;

        private int MaxTokens { get; set; } = 256;

        private readonly AIRequestSettings _chatRequestSettings;

        public CollectPreferredLanguageToolPlugin(IKernel kernel)
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
        [Description("This function is used to ask user the name of the programming language they intend to use.")]
        [SKName("CollectPreferredLanguageTool")]
        public async Task<string> CollectPreferredLanguageToolAsync(
            [SKName("programming_language")][Description("The programming language the user intends to use")] string programming_language,
            SKContext context)
        {
            //Console.WriteLine("<======= Creating CollectPreferredLanguageTool chat =======>\n");
            var chat = this._chat.CreateNewChat(SystemPrompt);
            chat.AddUserMessage(Goal);

            ChatHistory? chatHistory = context.GetChatHistory();
            if (chatHistory?.Any() ?? false)
            {
                chat.Messages.AddRange(chatHistory);
            }

            //Console.WriteLine("000");
            if (!string.IsNullOrEmpty(programming_language))
            {
                //  Console.WriteLine($"111 {programming_language} programming_language");
                context.Variables["programming_language"] = programming_language;
                return programming_language;
            }
            //Console.WriteLine($"222 {programming_language} programming_language");

            context.PromptInput();
            //Console.WriteLine($"333 {programming_language} programming_language");

            return await this._chat.GenerateMessageAsync(chat, this._chatRequestSettings).ConfigureAwait(false);
        }
    }
    public sealed class GenerateFunctionSignaturePlugin
    {
        private const string ProgrammingLanguage = "programming_language";

        private const string Goal = "Generate function signature.";

        private const string SystemPrompt =
            @$"Based on the {ProgrammingLanguage} given by the user, generate a function signature framework the user should use to implement the solution.
The function signature framework you provide to the user should look something like this if the {ProgrammingLanguage} the user provided is in python:
[function signature]
    def max_subarray_sum(arr):
        """"""
        This function takes an array of integers as input and returns the maximum sum of any contiguous subarray.

        Args:
        arr (List[int]): A list of integers

        Returns:
        int: The maximum sum of any contiguous subarray
        """"""
        pass
[END function signature]
Provide the function signature to the user.";
        private readonly IChatCompletion _chat;

        private int MaxTokens { get; set; } = 256;

        private readonly AIRequestSettings _chatRequestSettings;

        public GenerateFunctionSignaturePlugin(IKernel kernel)
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
        [Description("This function is used to generate the function signature the user should use to implement the solution")]
        [SKName("GenerateFunctionSignature")]
        public async Task<string> GenerateFunctionSignatureAsync(
            [SKName("programming_language")][Description("The programming language user wants to use")] string programming_language,
            [SKName("function_signature")][Description("The function signature generated based on user's programming language")] string function_signature,
            SKContext context)
        {
            //Console.WriteLine("<======= Creating GenerateFunctionSignature chat =======>\n");
            var chat = this._chat.CreateNewChat(SystemPrompt);
            chat.AddUserMessage(Goal);

            ChatHistory? chatHistory = context.GetChatHistory();
            if (chatHistory?.Any() ?? false)
            {
                chat.Messages.AddRange(chatHistory);
            }

            if (!string.IsNullOrEmpty(function_signature))
            {
                context.Variables["function_signature"] = function_signature;
                Console.WriteLine("Assistant: Here's a function signature you could use to implement your final solution: \n" + function_signature);
                return "Assistant: Here's a function signature you could use to implement your final solution: \n" + function_signature;
            }

            return "Assistant: Here's a function signature you could use to implement your final solution: \n" + await this._chat.GenerateMessageAsync(chat, this._chatRequestSettings).ConfigureAwait(false);
        }
    }
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
    public sealed class PromptTimeAndSpaceComplexityAnalysisPlugin
    {
        private const string Problem = "problem_statement";
        private const string Solution = "_solution_code_implementation";
        private const string Delimiter = "```";
        private const string Goal = "Ask user to analyze the time and space complexity of their final solution's code implementation.";

        private const string SystemPrompt =
            @$"[Instruction]
The user has provided the final {Solution} to the {Problem}.
Your only task is to get the time and space complexity analysis from the user.
You can only ask user to analyze the time and space complexity based on their solution.
If the user responds with anything other than a time complexity or a space complexity, say that you don't know.
You are not allowed to give the analysis to the user.

The last step is to provide the user with a JSON object with time complexity and the space complexity. This object will be of the format [JSON format].
You will reply to the user with this JSON object AFTER they confirm their final time complexity and space complexity.
IMPORTANT: You will reply with the JSON object ONLY. This object will be wrapped in the delimiter {Delimiter}. DO NOT INCLUDE ANY OTHER WORDS OR EXPLANATIONS.
[End Instruction]

[JSON format]
{{
    ""timeComplexity"": <time complexity>,
    ""spaceComplexity"": <space complexity>
}}
[END JSON format]

[example]
    User: I think the time complexity is O(n) since I iterate through the array exactly once and the space complexity is O(1) since there's no extra space used.
    Assistant: That's right. Your solution has an optimal time and space complexity. Can you confirm your final time complexity is O(n) and space complexity is O(1)?
    User: Yes.
    Assistant:
        {Delimiter}
        {{
            ""timeComplexity"": ""O(n)"",
            ""spaceComplexity"": ""O1)""
        }}
        {Delimiter}
[END example]
";

        private readonly IChatCompletion _chat;

        private int MaxTokens { get; set; } = 1024;

        private readonly AIRequestSettings _chatRequestSettings;

        public PromptTimeAndSpaceComplexityAnalysisPlugin(IKernel kernel)
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
        [Description("This function is used to get from user the time and space complexity to the coding problem")]
        [SKName("PromptComplexity")]
        public async Task<string> PromptComplexityAsync(
            [SKName("problem_statement")][Description("The problem statement generated by the interviewer")] string problem_statement,
            SKContext context)
        {
            //Console.WriteLine("<======= Creating PromptComplexity chat =======>\n");
            var chat = this._chat.CreateNewChat(SystemPrompt);
            chat.AddUserMessage(Goal);

            ChatHistory? chatHistory = context.GetChatHistory();
            if (chatHistory?.Any() ?? false)
            {
                chat.Messages.AddRange(chatHistory);
            }

            context.Variables.TryGetValue("_solution_code_implementation", out string solution);
            context.Variables["_solution_code_implementation"] = solution;

            var response = await this._chat.GenerateMessageAsync(chat, this._chatRequestSettings).ConfigureAwait(false);

            var jsonRegex = new Regex($"{Delimiter}\\s*({{.*}})\\s*{Delimiter}", RegexOptions.Singleline);
            var match = jsonRegex.Match(response);

            if (match.Success)
            {
                var json = match.Groups[1].Value;
                var complexityJson = JsonConvert.DeserializeObject<JObject>(json);

                context.Variables["time_complexity"] = complexityJson["timeComplexity"].Value<string>();
                context.Variables["space_complexity"] = complexityJson["spaceComplexity"].Value<string>();

                //Console.WriteLine("222222222222 complexities were successful 2222222222222");
                // Since we're not prompting input and solution is obtained, this won't be added to the messages
                return "User has provided their time and space complexity analysis.";
            }
            else
            {
                //Console.WriteLine("333333333333 prompt complexities again 333333333333333");

                context.PromptInput();
                return response;
            }
        }
    }

    public sealed class ConcludeInterviewPlugin
    {
        private const string Goal = "As the coding interviewer, give user feedback and a decision based on user's interview performance.";

        private const string Solution = "_solution_code_implementation";

        private const string TimeComplexity = "time_complexity";

        private const string SpaceComplexity = "space_complexity";

        private const string SystemPrompt =
    @$"Steps to follow:
1. Analyze user's interview in the previous conversation
based on user's coding interview solution: {Solution},
time complexity {TimeComplexity}, and space complexity {SpaceComplexity}.
2. Score user's performance and give user 3 scores out of 10 and 1 score out of 5 for test case walk-through:
    - code reliabiity,
    - code readability,
    - time and space optimality,
    - test case walk-through or a slack thereof.
3. Tally up total scores out of 35. Tell user they passed the interview if above 27.
Otherwise, tell them they failed and can retake after a month.
4. Summarize user's interview performance and give user justification for each scoring criteria..
5. End this task and the interview.

[Example]
User: How did I do on the interview?
Assistant:
    Firstly, your coding solution is correct and efficient with a time complexity of O(n). This shows that you have a good understanding of the problem and are able to come up with an optimal solution.
    In terms of code reliability, your solution is reliable and should work for most test cases. However, it is always good to consider edge cases and test your code thoroughly to ensure it is robust.
    Your code readability is good, as you have used clear variable names and have written concise code. However, you could improve the readability by adding comments to explain your thought process and the logic behind your code.
    Your communication score is also good, as you were able to explain your thought process and approach to the problem clearly. However, you could improve by asking clarifying questions and engaging in a dialogue with the interviewer.
    Testing is an essential part of software development. You could've done better with a walk-through of your implementation with some test cases
    Overall, I would give you a score of 8/10 for code reliability, 9/10 for code readability, 10/10 for time and space optimality, 1/5 for not running enough test cases.
    Congratulations! You passed the interview. I will be in touch to set up your next round of interview. Have a great day!
[End Example]
";

        private readonly IChatCompletion _chat;

        private int MaxTokens { get; set; } = 32768;

        private readonly AIRequestSettings _chatRequestSettings;

        public ConcludeInterviewPlugin(IKernel kernel)
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
        [Description("This function is used to give feedback on user's interview performance")]
        [SKName("GiveFeedback")]
        public async Task<string> GiveFeedbackAsync(
            [SKName("problem_statement")][Description("The problem given to user to solve")] string problem,
            [SKName("time_complexity")][Description("The time complexity of the user's final solution")] string time_complexity,
            [SKName("space_complexity")][Description("The space complexity of the user's final solution")] string space_complexity,
            [SKName("feedback")][Description("feedback for user's interview performance")] string feedback,
            [SKName("interview_decision")][Description("interview decision for user")] string interview_decision,
            SKContext context)
        {
            //Console.WriteLine("<======= Creating GiveFeedback chat =======>\n");

            context.Variables.TryGetValue("_solution_code_implementation", out string solution);

            context.Variables["_solution_code_implementation"] = solution;

            //Console.WriteLine($"5555555555 {context.Variables["_solution_code_implementation"]} 555555555");
            var chat = this._chat.CreateNewChat(SystemPrompt);
            chat.AddUserMessage(Goal);

            ChatHistory? chatHistory = context.GetChatHistory();
            if (chatHistory?.Any() ?? false)
            {
                chat.Messages.AddRange(chatHistory);
            }

            var feedbackProvided = false;
            var interviewDecisionProvided = false;

            if (!string.IsNullOrEmpty(feedback))
            {
                context.Variables["feedback"] = feedback;
                feedbackProvided = true;
            }

            if (!string.IsNullOrEmpty(interview_decision))
            {
                context.Variables["interview_decision"] = interview_decision;
                interviewDecisionProvided = true;
            }

            if (feedbackProvided && interviewDecisionProvided)
            {
                context.PromptInput();
                return "Assistant: " + context.Variables["feedback"]
                     + "\nDecision is: " + context.Variables["interview_decision"];
            }

            return "Assistant: " + await this._chat.GenerateMessageAsync(chat, this._chatRequestSettings).ConfigureAwait(false);
        }
    }
}

//Running Example64_FlowOrchestrator...
//*****************************************************
//Flow: FlowOrchestrator_Example_Flow
//Assistant: Hello! Thanks for joining the coding interview. Here's the problem for you to solve:
//Write a function that takes in a list of integers and returns the maximum sum of any contiguous subarray within the list.
//Assistant: ["Sure, what programming language do you plan to use to solve your problem?"]

//User: python
//Assistant: Here's a function signature you could use to implement your final solution:
//def max_subarray_sum(arr: List[int]) -> int:
//Assistant: ["How would you like to implement the solution to find the maximum subarray sum problem? Please provide your code implementation in your preferred programming language. Remember to include the function signature provided earlier."]

//User: Are there any constraints on the size of the input and can I assume the input will always be valid?
//Assistant: ["There are no constraints on the size of the input and you can assume that the input will always be valid."]

//User: Do I need to implement in real code or just pseudo code is enough?
//Assistant: ["Please provide the actual code implementation rather than pseudo code."]

//User: I can loop through all possible subarrays of the input list but that might not be optimal.
//Assistant: ["You are correct. Looping through all possible subarrays of the input list is a valid approach to solve the problem of finding the maximum subarray sum. However, it has a time complexity of O(n^3) which is not efficient and may not be suitable for large input sizes. Can you think of a way to optimize it?"]

//User: Well, I also observe that the maximum sum of a contiguous subarray ending at index i depends on the maximum sum of a contiguous subarray ending at index i-1.
//Assistant: ["That\u0027s a great observation! Can you explain how you can use this observation to optimize the approach?"]

//User: Can I start with the pseudo code?
//Assistant: ["Sure, you can start with the pseudo code."]

//User: I will initialize max_sum and current_sum to the first element of the list. Then, I will loop through the list, adding each element to current_sum. If current_sum is greater than max_sum, I will update max_sum. If current_sum is negative, I will reset it to zero. After processing all elements, I will return max_sum, which represents the maximum sum of any contiguous subarray within the list.
//Assistant: ["You are on the right track. Can you go ahead and implement the solution in your preferred programming language?"]

//User: My solution is:
//def max_subarray_sum(arr):
// n = len(arr)
// max_sum = arr[0]
// current_sum = arr[0]
//    for i in range(1, n):
//        current_sum = max(arr[i], current_sum + arr[i])
//        max_sum = max(max_sum, current_sum)
//    return max_sum
//Assistant: ["Great! Your solution looks good to me. Is this your final code implementation?"]

//User: Yes.
//Assistant: ["What is the time complexity and space complexity of your final solution\u0027s code implementation?"]

//User: I think the time complexity is O(n) since I iterate through the array exactly once and the space complexity is O(1) since there's no extra space used.//Assistant: Feedback provided:
//Assistant: You did well in explaining your thought process and arriving at a solution with optimal time and space complexity. However, you could have provided more test cases to validate your solution.
//Decision is: Hire
