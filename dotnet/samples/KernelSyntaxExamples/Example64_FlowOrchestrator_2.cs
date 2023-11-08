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

/**
 * This example shows how to use FlowOrchestrator to execute a given flow with interaction with client.
 */

// ReSharper disable once InconsistentNaming
public static class Example62_FlowOrchestrator
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
      - goal: Clarify problem constraint questions user asks.
        plugins:
          - ClarifyConstraintsPlugin
        requires:
          - problem_statement
      - goal: Ask user to code the problem solution. Get fully implemented solution from user.
        plugins:
          - PromptSolutionPlugin
        requires:
          - problem_statement
          - programming_language
          - function_signature
        provides:
          - solution
      - goal: Ask user to analyze time and space complexity of their final solution. Score user's interview performance based on the communication, code reliability, code readability.
        plugins:
          - GiveFeedbackPlugin
          - TimeComplexityAnalysisPlugin
        requires:
          - problem_statement
          - solution
        provides:
          - feedback
          - time_complexity
          - space_complexity
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
                .AddFilter(null, LogLevel.Warning));

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

        Console.WriteLine("Interviewer: " + result.ToString());

        string[] userInputs = new[]
        {
            "python",
            "Are there any constraints on the size of the input and can I assume the input will always be valid?",
            "Do I need to implement the whole solution or just pseudo code is enough?",
            "I can loop through all possible subarrays of the input list but that might not be optimal.",
            "Well, I also observe that the maximum sum of a contiguous subarray ending at index i depends on the maximum sum of a contiguous subarray ending at index i-1.",
            "Can I start with the pseudo code?",
            "I will initialize max_sum and current_sum to the first element of the list. Then, I will loop through the list, adding each element to current_sum. If current_sum is greater than max_sum, I will update max_sum. If current_sum is negative, I will reset it to zero. After processing all elements, I will return max_sum, which represents the maximum sum of any contiguous subarray within the list.",
            @"{""solution"" : ""def max_subarray_sum(arr):\n n = len(arr)\n max_sum = arr[0]\n current_sum = arr[0]\n    for i in range(1, n):\n        current_sum = max(arr[i], current_sum + arr[i])\n        max_sum = max(max_sum, current_sum)\n    return max_sum""}",
            "Yes.",
            "I think the time complexity is O(n) since I iterate through the array exactly once and the space complexity is O(1) since there's no extra space used.",
        };

        foreach (var t in userInputs)
        {
            Console.WriteLine($"\nUser: {t}");
            result = await orchestrator.ExecuteFlowAsync(s_flow, sessionId, t).ConfigureAwait(false);
            Console.WriteLine("Interviewer: " + result.ToString());
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
            //.WithLoggerFactory(loggerFactory);
            .WithLoggerFactory(
                LoggerFactory.Create(option => option.AddConsole()));
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

                Console.WriteLine("Interviewer: Hello! Thanks for joining the coding interview. " +
                   "Here's the problem for you to solve: \n" + problem);

                return "Interviewer: Hello! Thanks for joining the coding interview. " +
                   "Here's the problem for you to solve: \n" + problem;
            }

            return "Interviewer: Hello! Thanks for joining the coding interview. " +
                   "Here's the problem for you to solve: \n" +
                   await this._chat.GenerateMessageAsync(chat, this._chatRequestSettings).ConfigureAwait(false);
        }
    }

    public sealed class ClarifyConstraintsPlugin
    {
        private const string Goal = "Clarify problem constraints";

        private const string ProblemStatement = "problem_statement";

        private const string SystemPrompt =
            $@"
[instructions
Based on the {ProblemStatement} provided to the user earlier,
Answer a maximum of 3 questions by keeping track of the NumQuestionsAsked variable.
You can answer user's questions only related to problem constraints such as:
    - input size,
    - input validity
You are not allowed to tell user how to solve the problem or provide code logic.
Answer questions but not engage in problem discussions or user's attempt at analyze problem approach.
[END instructions]
User: What's the input size?
Interviewer: It's big and your solution should take that into consideration and not time out.
[example conversation]

[example user asks about how to solve the problem]
User: I can iterate through the array but that might be too time consuming.
Interviewer: ""
";
        private int NumQuestionsAsked = 0;

        private readonly IChatCompletion _chat;

        private int MaxTokens { get; set; } = 256;

        private readonly AIRequestSettings _chatRequestSettings;

        public ClarifyConstraintsPlugin(IKernel kernel)
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
        [Description("This function is used to answer user's questions related to problem constraints.")]
        [SKName("ClarifyConstraints")]
        public async Task<string> ClarifyConstraintsAsync(
            [SKName("problem_statement")][Description("The coding problem generated by the assistant")] string problem,
            SKContext context)
        {
            Console.WriteLine("<======= Creating ClarifyConstraintsPlugin chat =======>\n");

            var chat = this._chat.CreateNewChat(SystemPrompt);
            chat.AddUserMessage(Goal);

            ChatHistory? chatHistory = context.GetChatHistory();
            if (chatHistory?.Any() ?? false)
            {
                chat.Messages.AddRange(chatHistory);
            }

            if (NumQuestionsAsked < 3)
            {
                context.PromptInput();
                NumQuestionsAsked += 1;
            }

            return await this._chat.GenerateMessageAsync(chat, this._chatRequestSettings).ConfigureAwait(false);
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
                Console.WriteLine("Interviewer: Here's a function signature you could use to implement your final solution: \n" + function_signature);
                return "Interviewer: Here's a function signature you could use to implement your final solution: \n" + function_signature;
            }

            return "Interviewer: Here's a function signature you could use to implement your final solution: \n" + await this._chat.GenerateMessageAsync(chat, this._chatRequestSettings).ConfigureAwait(false);
        }
    }
    public sealed class PromptSolutionPlugin
    {
        private const string Problem = "problem_statement";
        private const string ProgrammingLanguage = "programming_language";
        private const string FunctionSignature = "function_signature";
        private const string Delimiter = "```";
        private const string Goal = "Ask user for the solution to the problem.";

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
- If the solution logic proposed by the user is a brute force solution, prompt the user to optimize the solution.
- Prompt user to implement the most optimal solution using their preferred {ProgrammingLanguage}

The last step is to provide the user with a JSO object with user's solution code implemented in {ProgrammingLanguage}. This object will be of the format [JSON format].
You will reply to the user with this JSON object AFTER they confirm their final solution.

IMPORTANT: Again, you cannot write any solution code for the user!
IMPORTANT: You cannot tell user the time complexity of their solution
IMPORTANT: You only need to get the full implementation from the user. You don't need to validate the solution.
[END Instruction]

[Rules]
    Rule 1: The solution must be provided by the user.
    Rule 2: The solution must contain the {FunctionSignature} the interviewer previously generated.
    Rule 3: The solution must be implemented in the {ProgrammingLanguage} the user previously provided.

[JSON format]
{{
    ""solution"": <string>
}}.
[END JSON format]

[Example]
    Interviewer: How do you want to approach the problem?
    User: A brute force solution I can think of is to loop through all possible subarrays of the input list.
    Interviewer: This is a valid approach to solve the problem of finding the maximum subarray sum. However, it has a time complexity of O(n^3) which is not efficient and may not be suitable for large input sizes. Can you think of a way to optimize it?
    User: Well, I also observe that the maximum sum of a contiguous subarray ending at index i depends on the maximum sum of a contiguous subarray ending at index i-1.
    Interviewer: That's a great observation! Can you explain how you can use this observation to optimize the solution?
    User: I will initialize max_sum and current_sum to the first element of the list. Then, I will loop through the list, adding each element to current_sum. If current_sum is greater than max_sum, I will update max_sum. If current_sum is negative, I will reset it to zero. After processing all elements, I will return max_sum, which represents the maximum sum of any contiguous subarray within the list.
    Interviewer: You are on the right track. Can you go aheaed and implement the solution in your preferred programming language?
    User:
        def max_subarray_sum(arr):
            n = len(arr)
            max_sum = arr[0]
            current_sum = arr[0]
            for i in range(1, n):
                current_sum = max(arr[i], current_sum + arr[i])
                max_sum = max(max_sum, current_sum)
            return max_sum
    Interviewer: Is this your final solution?
    User: Yes
    Interviewer:
    {Delimiter}
    {{
        ""solution"": ""def max_subarray_sum(arr):
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
        [Description("This function is used to get from user the solution to the coding problem")]
        [SKName("PromptSolution")]
        public async Task<string> PromptSolutionAsync(
            [SKName("problem_statement")][Description("The problem statement generated by the assistant")] string problem_statement,
            [SKName("programming_language")][Description("The preferred programming language the user intends to use for the interview given by the user")] string programming_language,
            [SKName("function_signature")][Description("The function signature generated by the assistant")] string function_signature,
            SKContext context)
        {
            Console.WriteLine("<======= Creating PromptSolution chat =======>\n");
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
                var solution_json = JsonConvert.DeserializeObject<JObject>(json);

                context.Variables["solution"] = solution_json["solution"].Value<string>();

                // Since we're not prompting input and solution is obtained, this won't be added to the messages
                return "Done";
            }
            else
            {
                context.PromptInput();
                return response;
            }

        }
    }
    public sealed class TimeComplexityAnalysisPlugin
    {
        private const string Problem = "problem_statement";
        private const string Solution = "solution";
        private const string Goal = "Ask user to analyze the time and space complexity of their final solution.";

        private const string SystemPrompt =
            @$"The user has provided the final {Solution} to the {Problem}.
If the user already provided the analysis, comment on the analysis and end the task.
Ask user to to analyze the time and space complexity of their final solution.
You are not allowed to give the analysis to the user.
If the user says anything unrelated to the time and space complexity analysis, say that you don't know.

[example user already provdied time and space complexity analysis without prompting]
User: I think the time complexity is O(n) and Space complexity is O(1).
Interviewer: That's correct (That's not correct).

You only need to get the time and space complexity from the user.";
        private readonly IChatCompletion _chat;

        private int MaxTokens { get; set; } = 1024;

        private readonly AIRequestSettings _chatRequestSettings;

        public TimeComplexityAnalysisPlugin(IKernel kernel)
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
        [SKName("TimeComplexityAnalysis")]
        public async Task<string> TimeComplexityAnalysisAsync(
            [SKName("problem_statement")][Description("The problem statement generated by the assistant")] string problem_statement,
            [SKName("solution")][Description("The full code implementation given by the user")] string solution,
            [SKName("time_complexity")][Description("The time complexity of the user's final solution")] string time_complexity,
            [SKName("space_complexity")][Description("The space complexity of the user's final solution")] string space_complexity,
            SKContext context)
        {
            Console.WriteLine("<======= Creating TimeComplexityAnalysis chat =======>\n");
            var chat = this._chat.CreateNewChat(SystemPrompt);
            chat.AddUserMessage(Goal);

            ChatHistory? chatHistory = context.GetChatHistory();
            if (chatHistory?.Any() ?? false)
            {
                chat.Messages.AddRange(chatHistory);
            }

            if (!string.IsNullOrEmpty(time_complexity))
            {
                context.Variables["time_complexity"] = time_complexity;
                return "You have provided the time complexity: " + time_complexity;
            }

            if (!string.IsNullOrEmpty(space_complexity))
            {
                context.Variables["space_complexity"] = space_complexity;
                return "You have provided the space complexity: " + space_complexity;
            }

            context.PromptInput();
            return await this._chat.GenerateMessageAsync(chat, this._chatRequestSettings).ConfigureAwait(false);
        }
    }

    public sealed class GiveFeedbackPlugin
    {
        private const string Goal = "As the coding interviewer, give user feedback and evaluate user's interview performance.";

        private const string Solution = "solution";

        private const string TimeComplexity = "time_complexity";

        private const string SpaceComplexity = "space_complexity";

        private const string SystemPrompt =
    @$"Steps to follow:
1. Analyze user's interview in the previous conversation
based on user's coding interview solution: {Solution},
time complexity {TimeComplexity}, and space complexity {SpaceComplexity}.
2. Score user's performance and give user 4 scores out of 10:
    - code reliabiity,
    - code readability,
    - communication score,
    - test case walk-through or a slack thereof.
3. Tally up total scores out of 40. Tell user they passed the interview if above 32.
Otherwise, tell them they failed and can retake after a month.
4. Summarize user's interview performance and give user justification for each scoring criteria..
5. End this task and the interview.";

        private readonly IChatCompletion _chat;

        private int MaxTokens { get; set; } = 8192;

        private readonly AIRequestSettings _chatRequestSettings;

        public GiveFeedbackPlugin(IKernel kernel)
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
        [Description("This function is used to evaluate user's interview performance")]
        [SKName("GiveFeedback")]
        public async Task<string> GiveFeedbackAsync(
            [SKName("solution")][Description("The full implementation given by the user")] string solution,
            [SKName("time_complexity")][Description("The time complexity of the user's final solution")] string time_complexity,
            [SKName("feedback")][Description("feedback for user's interview performance")] string feedback,
            SKContext context)
        {
            Console.WriteLine("<======= Creating GiveFeedbackPlugin chat =======>\n");

            var chat = this._chat.CreateNewChat(SystemPrompt);
            chat.AddUserMessage(Goal);

            ChatHistory? chatHistory = context.GetChatHistory();
            if (chatHistory?.Any() ?? false)
            {
                chat.Messages.AddRange(chatHistory);
            }

            if (!string.IsNullOrEmpty(feedback))
            {
                context.Variables["feedback"] = feedback;
                return $"Interviewer: {feedback}";
            }

            context.PromptInput();
            return await this._chat.GenerateMessageAsync(chat, this._chatRequestSettings).ConfigureAwait(false);
        }
    }
}

// solution gets parsed correctly with the @ ""solution""
//// Copyright (c) Microsoft. All rights reserved.

//using System;
//using System.Collections.Generic;
//using System.ComponentModel;
//using System.Diagnostics;
//using System.Linq;
//using System.Text;
//using System.Text.Json;
//using System.Text.RegularExpressions;
//using System.Threading.Tasks;
//using Azure.Search.Documents.Models;
//using Google.Apis.CustomSearchAPI.v1.Data;
//using Kusto.Cloud.Platform.Utils;
//using Kusto.Data.Common;
//using Microsoft.Extensions.Logging;
//using Microsoft.SemanticKernel;
//using Microsoft.SemanticKernel.AI;
//using Microsoft.SemanticKernel.AI.ChatCompletion;
//using Microsoft.SemanticKernel.Connectors.AI.OpenAI;
//using Microsoft.SemanticKernel.Experimental.Orchestration;
//using Microsoft.SemanticKernel.Orchestration;
//using Microsoft.SemanticKernel.Plugins.Core;
//using Microsoft.SemanticKernel.Plugins.Memory;
//using Microsoft.SemanticKernel.Plugins.Web;
//using Microsoft.SemanticKernel.Plugins.Web.Bing;
//using Microsoft.SemanticKernel.TemplateEngine.Basic;
//using NRedisStack.Graph.DataTypes;
//using StackExchange.Redis;

///**
// * This example shows how to use FlowOrchestrator to execute a given flow with interaction with client.
// */

//// ReSharper disable once InconsistentNaming
//public static class Example62_FlowOrchestrator
//{
//    private static readonly Flow s_flow = FlowSerializer.DeserializeFromYaml(@"
//    name: FlowOrchestrator_Example_Flow
//    goal: generate interview problem and conduct coding interview with user
//    steps:
//      - goal: Generate the coding problem prompt that outputs the maximum subarray in a list.
//        plugins:
//          - GenerateProblemPlugin
//        provides:
//          - problem_statement
//      - goal: Ask user what programming language they will use to implement the solution. Generate the function signature they should implement the solution in.
//        plugins:
//          - CollectPreferredLanguageToolPlugin
//          - GenerateFunctionSignaturePlugin
//        requires:
//          - problem_statement
//        provides:
//          - programming_language
//          - function_signature
//      - goal: Ask user to code the problem solution. Get fully implemented solution from user.
//        plugins:
//          - PromptSolutionPlugin
//        requires:
//          - problem_statement
//          - programming_language
//          - function_signature
//        provides:
//          - solution
//      - goal: Ask user to anayze time and space complexity of their final solution.
//        plugins:
//          - TimeComplexityAnalysisPlugin
//        requires:
//          - solution
//        provides:
//          - time_complexity
//          - space_complexity
//      - goal: Score user's interview performance based on the communication, code reliability, code readability.
//        plugins:
//          - GiveFeedbackPlugin
//        requires:
//          - problem_statement
//          - solution
//          - time_complexity
//          - space_complexity
//        provides:
//          - score
//");
//    public static Task RunAsync()
//    {
//        return RunExampleAsync();
//    }

//    private static async Task RunExampleAsync()
//    {
//        var bingConnector = new BingConnector(TestConfiguration.Bing.ApiKey);
//        var webSearchEnginePlugin = new WebSearchEnginePlugin(bingConnector);
//        using var loggerFactory = LoggerFactory.Create(loggerBuilder =>
//            loggerBuilder
//                .AddConsole()
//                .AddFilter(null, LogLevel.Warning));

//        Dictionary<object, string?> plugins = new()
//        {
//            { webSearchEnginePlugin, "WebSearch" },
//        };
//        FlowOrchestrator orchestrator = new(
//            GetKernelBuilder(loggerFactory),
//            await FlowStatusProvider.ConnectAsync(new VolatileMemoryStore()).ConfigureAwait(false),
//            plugins,
//            config: GetOrchestratorConfig());

//        var sessionId = Guid.NewGuid().ToString();

//        Console.WriteLine("*****************************************************");
//        Stopwatch sw = new();
//        sw.Start();
//        Console.WriteLine("Flow: " + s_flow.Name);
//        var result = await orchestrator.ExecuteFlowAsync(s_flow, sessionId, "Execute the flow").ConfigureAwait(false);

//        Console.WriteLine("Interviewer: " + result.ToString());

//        string[] userInputs = new[]
//        {
//            "python",
//            "Are there any constraints on the size of the input?",
//            "Can I assume the input will always be valid?",
//            "Do I need to implement the whole solution or just pseudo code is enough?",
//            "I can loop through all possible subarrays of the input list but that might not be optimal.",
//            "Well, I also observe that the maximum sum of a contiguous subarray ending at index i depends on the maximum sum of a contiguous subarray ending at index i-1.",
//            "I will initialize max_sum and current_sum to the first element of the list. Then, I will loop through the list, adding each element to current_sum. If current_sum is greater than max_sum, I will update max_sum. If current_sum is negative, I will reset it to zero. After processing all elements, I will return max_sum, which represents the maximum sum of any contiguous subarray within the list.",
//            @"{""solution"" : ""def max_subarray_sum(arr):\n n = len(arr)\n max_sum = arr[0]\n current_sum = arr[0]\n    for i in range(1, n):\n        current_sum = max(arr[i], current_sum + arr[i])\n        max_sum = max(max_sum, current_sum)\n    return max_sum""}",
//            "Should I analyze the time and space complexity next?",
//            "I think the time complexity is O(n) since I iterate through the array exactly once.",
//            "And the space complexity is O(1) since there's no extra space used.",
//        };

//        foreach (var t in userInputs)
//        {
//            Console.WriteLine($"\nUser: {t}");
//            result = await orchestrator.ExecuteFlowAsync(s_flow, sessionId, t).ConfigureAwait(false);
//            Console.WriteLine("Interviewer: " + result.ToString());
//        }

//        Console.WriteLine("Time Taken: " + sw.Elapsed);
//        Console.WriteLine("*****************************************************");
//    }
//    private static FlowOrchestratorConfig GetOrchestratorConfig()
//    {
//        var config = new FlowOrchestratorConfig
//        {
//            ReActModel = FlowOrchestratorConfig.ModelName.GPT35_TURBO,
//            MaxStepIterations = 20
//        };

//        return config;
//    }
//    private static KernelBuilder GetKernelBuilder(ILoggerFactory loggerFactory)
//    {
//        var builder = new KernelBuilder();

//        return builder
//            .WithAzureChatCompletionService(
//                TestConfiguration.AzureOpenAI.ChatDeploymentName,
//                TestConfiguration.AzureOpenAI.Endpoint,
//                TestConfiguration.AzureOpenAI.ApiKey,
//                true,
//                setAsDefault: true)
//            .WithRetryBasic(new()
//            {
//                MaxRetryCount = 3,
//                UseExponentialBackoff = true,
//                MinRetryDelay = TimeSpan.FromSeconds(3),
//            })
//            .WithPromptTemplateEngine(new BasicPromptTemplateEngine(loggerFactory))
//            .WithLoggerFactory(
//                LoggerFactory.Create(option => option.AddConsole()));
//    }

//    public sealed class GenerateProblemPlugin
//    {
//        private const string Goal = "Generate coding problem prompt of finding maximum subarray in a list";
//        private const string SystemPrompt =
//            "I am a question generating bot. I will describe the coding problem " +
//            "of finding maximum subarray in a list for the user to solve.";

//        private readonly IChatCompletion _chat;

//        private int MaxTokens { get; set; } = 256;

//        private readonly AIRequestSettings _chatRequestSettings;

//        public GenerateProblemPlugin(IKernel kernel)
//        {
//            this._chat = kernel.GetService<IChatCompletion>();
//            this._chatRequestSettings = new OpenAIRequestSettings
//            {
//                MaxTokens = this.MaxTokens,
//                StopSequences = new List<string>() { "Observation:" },
//                Temperature = 0
//            };
//        }

//        [SKFunction]
//        [Description("This function is used to generate a coding problem")]
//        [SKName("GenerateProblem")]
//        public async Task<string> GenerateProblemAsync(
//            [SKName("problem_statement")][Description("The coding problem prompt")] string problem,
//            SKContext context)
//        {
//            Console.WriteLine("<======= Creating GenerateProblem chat =======>\n");
//            var chat = this._chat.CreateNewChat(SystemPrompt);
//            chat.AddUserMessage(Goal);

//            ChatHistory? chatHistory = context.GetChatHistory();
//            if (chatHistory?.Any() ?? false)
//            {
//                chat.Messages.AddRange(chatHistory);
//            }

//            if (!string.IsNullOrEmpty(problem))
//            {
//                context.Variables["problem_statement"] = problem;

//                Console.WriteLine("Interviewer: Hello! Thanks for joining the coding interview. " +
//                   "Here's the problem for you to solve: \n" + problem);

//                return "Interviewer: Hello! Thanks for joining the coding interview. " +
//                   "Here's the problem for you to solve: \n" + problem;
//            }

//            return "Interviewer: Hello! Thanks for joining the coding interview. " +
//                   "Here's the problem for you to solve: \n" +
//                   await this._chat.GenerateMessageAsync(chat, this._chatRequestSettings).ConfigureAwait(false);
//        }
//    }

//    public sealed class PromptUserPlugin
//    {
//        private const string Goal = "You have generated a coding problem. Now answer user's questions until user is ready to solve the problem.";

//        private const string SystemPrompt =
//            "You are an online coding interviewer. You have provided a coding problem for the user to solve. " +
//            "If the user asks a question regarding logistics of the interview, answer them. " +
//            "Otherwise, do not provide extra information the user does not ask for. " +
//            "For example, you can answer questions like the what is the interview process like " +
//            "can I use any language to write the solution " +
//            "or how big is the input " +
//            "but not questions directly related to solving the problem. " +
//            "Ask if user has any more questions.";
//        private readonly IChatCompletion _chat;

//        private int MaxTokens { get; set; } = 256;

//        private readonly AIRequestSettings _chatRequestSettings;

//        public PromptUserPlugin(IKernel kernel)
//        {
//            this._chat = kernel.GetService<IChatCompletion>();
//            this._chatRequestSettings = new OpenAIRequestSettings
//            {
//                MaxTokens = this.MaxTokens,
//                StopSequences = new List<string>() { "Observation:" },
//                Temperature = 0
//            };
//        }

//        [SKFunction]
//        [Description("This function is used to ask if user is ready to start the interview.")]
//        [SKName("PromptUser")]
//        public async Task<string> PromptUserAsync(
//            [SKName("problem_statement")][Description("The coding problem generated by the assistant")] string problem,
//            SKContext context)
//        {
//            Console.WriteLine("<======= Creating PromptUserPlugin chat =======>\n");

//            var chat = this._chat.CreateNewChat(SystemPrompt);
//            chat.AddUserMessage(Goal);

//            //Console.WriteLine("Recall stated problem is: \n" + context.Variables["problem_statement"] + "\n");

//            ChatHistory? chatHistory = context.GetChatHistory();
//            if (chatHistory?.Any() ?? false)
//            {
//                chat.Messages.AddRange(chatHistory);
//            }

//            context.PromptInput();

//            return await this._chat.GenerateMessageAsync(chat, this._chatRequestSettings).ConfigureAwait(false);
//        }
//    }
//    public sealed class CollectPreferredLanguageToolPlugin
//    {
//        private const string Goal = "Get programming language from user.";

//        private const string SystemPrompt =
//            "Your only task is to get the name of the programming language the user intends to use. " +
//            "You can only ask user what programming language they will use to solve the given problem. " +
//            "You cannot respond to input from the user about solving the problem. " +
//            "You cannot solve the problem for the user or provide any hints or snippets of code. " +
//            "If the user responds with anything other than a programming language, say that you don't know. " +
//            "You cannot answer questions that will give the user the entire logic to the problem. " +
//            "You cannot write any solution code for the user. " +
//            "You cannot explain the problem or solution to the user.";
//        private readonly IChatCompletion _chat;

//        private int MaxTokens { get; set; } = 256;

//        private readonly AIRequestSettings _chatRequestSettings;

//        public CollectPreferredLanguageToolPlugin(IKernel kernel)
//        {
//            this._chat = kernel.GetService<IChatCompletion>();
//            this._chatRequestSettings = new OpenAIRequestSettings
//            {
//                MaxTokens = this.MaxTokens,
//                StopSequences = new List<string>() { "Observation:" },
//                Temperature = 0
//            };
//        }

//        [SKFunction]
//        [Description("This function is used to ask user the name of the programming language they intend to use.")]
//        [SKName("CollectPreferredLanguageTool")]
//        public async Task<string> CollectPreferredLanguageToolAsync(
//            [SKName("programming_language")][Description("The programming language the user intends to use")] string programming_language,
//            SKContext context)
//        {
//            Console.WriteLine("<======= Creating CollectPreferredLanguageTool chat =======>\n");
//            var chat = this._chat.CreateNewChat(SystemPrompt);
//            chat.AddUserMessage(Goal);

//            ChatHistory? chatHistory = context.GetChatHistory();
//            if (chatHistory?.Any() ?? false)
//            {
//                chat.Messages.AddRange(chatHistory);
//            }

//            Console.WriteLine("000");
//            if (!string.IsNullOrEmpty(programming_language))
//            {
//                Console.WriteLine($"111 {programming_language} programming_language");
//                context.Variables["programming_language"] = programming_language;
//                return programming_language;
//            }
//            Console.WriteLine($"222 {programming_language} programming_language");

//            context.PromptInput();
//            Console.WriteLine($"333 {programming_language} programming_language");

//            return await this._chat.GenerateMessageAsync(chat, this._chatRequestSettings).ConfigureAwait(false);
//        }
//    }
//    public sealed class GenerateFunctionSignaturePlugin
//    {
//        private const string ProgrammingLanguage = "programming_language";

//        private const string Goal = "Generate function signature.";

//        private const string SystemPrompt =
//            @$"Based on the {ProgrammingLanguage} given by the user, generate a function signature framework the user should use to implement the solution.
//The function signature framework you provide to the user should look something like this if the {ProgrammingLanguage} the user provided is in python:
//    def max_subarray_sum(arr):
//        """"""
//        This function takes an array of integers as input and returns the maximum sum of any contiguous subarray.

//        Args:
//        arr (List[int]): A list of integers

//        Returns:
//        int: The maximum sum of any contiguous subarray
//        """"""
//        pass
//Provide this function signature to the user.
//You
//";
//        private readonly IChatCompletion _chat;

//        private int MaxTokens { get; set; } = 256;

//        private readonly AIRequestSettings _chatRequestSettings;

//        public GenerateFunctionSignaturePlugin(IKernel kernel)
//        {
//            this._chat = kernel.GetService<IChatCompletion>();
//            this._chatRequestSettings = new OpenAIRequestSettings
//            {
//                MaxTokens = this.MaxTokens,
//                StopSequences = new List<string>() { "Observation:" },
//                Temperature = 0
//            };
//        }

//        [SKFunction]
//        [Description("This function is used to generate the function signature the user should use to implement the solution")]
//        [SKName("GenerateFunctionSignature")]
//        public async Task<string> GenerateFunctionSignatureAsync(
//            [SKName("programming_language")][Description("The programming language user wants to use")] string programming_language,
//            [SKName("function_signature")][Description("The function signature generated based on user's programming language")] string function_signature,
//            SKContext context)
//        {
//            Console.WriteLine("<======= Creating GenerateFunctionSignature chat =======>\n");
//            var chat = this._chat.CreateNewChat(SystemPrompt);
//            chat.AddUserMessage(Goal);

//            ChatHistory? chatHistory = context.GetChatHistory();
//            if (chatHistory?.Any() ?? false)
//            {
//                chat.Messages.AddRange(chatHistory);
//            }

//            if (!string.IsNullOrEmpty(function_signature))
//            {
//                context.Variables["function_signature"] = function_signature;
//                Console.WriteLine("Interviewer: " + function_signature);
//                return "Interviewer: " + function_signature;
//            }

//            return "Interviewer: " + await this._chat.GenerateMessageAsync(chat, this._chatRequestSettings).ConfigureAwait(false);
//        }
//    }
//    public sealed class PromptSolutionPlugin
//    {
//        private const string Problem = "problem_statement";
//        private const string ProgrammingLanguage = "programming_language";
//        private const string FunctionSignature = "function_signature";
//        private const string Goal = "Ask user for the solution to the problem.";

//        private const string SystemPrompt =
//            $@"
//[Note To Self]
//You are an online coding interviewer.
//You have provided the user with a {Problem} generated from the previous step to solve.
//You should be stingy with giving hints.
//You should not tell the user the code for the solution at all.
//You also shouldn't tell user the time complexity and space complexity.
//You cannot answer questions that will give the user the entire logic to the problem.
//You cannot write any solution code for the user.
//You cannot explain the problem or solution to the user.
//[End note To Self]

//Steps to follow are:
//- Ask the user how do they want to approach the problem.
//- Prompt the user to keep analyzing the problem rather than tell them what to do.
//- If the solution logic proposed by the user is a brute force solution, prompt the user to optimize the solution.
//- Prompt user to implement the most optimal solution using their preferred {ProgrammingLanguage} in the json format:
//- Make sure the final solution user provides follows below the JSON format.

//[example solution JSON format]
//{{
//    ""solution"": <string>
//}}.
//[END example solution JSON format]

//[Response for when user provides solution logic or code]
//User: A brute force solution I can think of is to loop through all possible subarrays of the input list.
//Interviewer: That's a start but can you think of a way to optimize this solution?
//[END Response for when user provides solution logic or code]

//Note:
//Again, you cannot write any solution code for the user!
//You cannot explain the problem or solution to the user!
//You cannot tell user the time complexity of their solution
//You only need to get the full implementation from the user. You don't need to validate the solution.

//The last step is to reply to the user with their solution JSON object after they provide the JSON object of their solution.";

//        private readonly IChatCompletion _chat;

//        private int MaxTokens { get; set; } = 256;

//        private readonly AIRequestSettings _chatRequestSettings;

//        public PromptSolutionPlugin(IKernel kernel)
//        {
//            this._chat = kernel.GetService<IChatCompletion>();
//            this._chatRequestSettings = new OpenAIRequestSettings
//            {
//                MaxTokens = this.MaxTokens,
//                StopSequences = new List<string>() { "Observation:" },
//                Temperature = 0
//            };
//        }

//        [SKFunction]
//        [Description("This function is used to get from user the solution to the coding problem")]
//        [SKName("PromptSolution")]
//        public async Task<string> PromptSolutionAsync(
//            [SKName("problem_statement")][Description("The problem statement generated by the assistant")] string problem_statement,
//            [SKName("programming_language")][Description("The preferred programming language the user intends to use for the interview given by the user")] string programming_language,
//            [SKName("function_signature")][Description("The function signature generated by the assistant")] string function_signature,
//            [SKName("solution")][Description("The full code implementation given by the user")] string solution,
//            SKContext context)
//        {
//            Console.WriteLine("<======= Creating PromptSolution chat =======>\n");
//            var chat = this._chat.CreateNewChat(SystemPrompt);
//            chat.AddUserMessage(Goal);

//            ChatHistory? chatHistory = context.GetChatHistory();
//            if (chatHistory?.Any() ?? false)
//            {
//                chat.Messages.AddRange(chatHistory);
//            }

//            Console.WriteLine($"----------- solution: {solution}");
//            if (!string.IsNullOrEmpty(solution))
//            {
//                Console.WriteLine("1000");
//                try
//                {
//                    var json = JsonSerializer.Deserialize<Dictionary<string, string>>(solution);
//                    Console.WriteLine($"contains key? {json.ContainsKey("solution")}");
//                    Console.WriteLine($"not empty? {!string.IsNullOrEmpty(json["solution"])}");
//                    if (json.ContainsKey("solution") && !string.IsNullOrEmpty(json["solution"]))
//                    {
//                        Console.WriteLine("2000");
//                        context.Variables["solution"] = json["solution"];
//                        Console.WriteLine("You have provided the final solution: \n" + json["solution"]);
//                        return "You have provided the final solution: \n" + json["solution"];
//                    }
//                    Console.WriteLine("3000");
//                }
//                catch (JsonException)
//                {
//                    Console.WriteLine("4000");
//                    Console.WriteLine($"Solution {solution} provided is not in a valid solution format.");
//                }
//            } else
//            {
//                Console.WriteLine("5000");
//            }

//            Console.WriteLine("6000");
//            Console.WriteLine($"Try again please. Solution {solution} provided is not in a valid solution format.");
//            context.Variables["solution"] = string.Empty;
//            context.PromptInput();

//            return await this._chat.GenerateMessageAsync(chat, this._chatRequestSettings).ConfigureAwait(false);
//        }
//    }
//    public sealed class TimeComplexityAnalysisPlugin
//    {
//        private const string Problem = "problem_statement";
//        private const string Solution = "solution";
//        private const string Goal = "Ask user to analyze the time and space complexity of their final solution.";

//        private const string SystemPrompt =
//            @$"The user has provided the final {Solution} to the {Problem}.
//Ask user to to analyze the time and space complexity of their final solution.
//You are not allowed to give the analysis to the user.
//If the user says anything unrelated to the time and space complexity analysis, say that you don't know.";
//        private readonly IChatCompletion _chat;

//        private int MaxTokens { get; set; } = 256;

//        private readonly AIRequestSettings _chatRequestSettings;

//        public TimeComplexityAnalysisPlugin(IKernel kernel)
//        {
//            this._chat = kernel.GetService<IChatCompletion>();
//            this._chatRequestSettings = new OpenAIRequestSettings
//            {
//                MaxTokens = this.MaxTokens,
//                StopSequences = new List<string>() { "Observation:" },
//                Temperature = 0
//            };
//        }

//        [SKFunction]
//        [Description("This function is used to get from user the time and space complexity to the coding problem")]
//        [SKName("TimeComplexityAnalysis")]
//        public async Task<string> TimeComplexityAnalysisAsync(
//            [SKName("problem_statement")][Description("The problem statement generated by the assistant")] string problem_statement,
//            [SKName("solution")][Description("The full code implementation given by the user")] string solution,
//            [SKName("time_complexity")][Description("The time complexity of the user's final solution")] string time_complexity,
//            [SKName("space_complexity")][Description("The space complexity of the user's final solution")] string space_complexity,
//            SKContext context)
//        {
//            Console.WriteLine("<======= Creating TimeComplexityAnalysis chat =======>\n");
//            var chat = this._chat.CreateNewChat(SystemPrompt);
//            chat.AddUserMessage(Goal);

//            ChatHistory? chatHistory = context.GetChatHistory();
//            if (chatHistory?.Any() ?? false)
//            {
//                chat.Messages.AddRange(chatHistory);
//            }

//            if (!string.IsNullOrEmpty(time_complexity))
//            {
//                context.Variables["time_complexity"] = time_complexity;
//                return "You have provided the time complexity: " + time_complexity;
//            }

//            if (!string.IsNullOrEmpty(space_complexity))
//            {
//                context.Variables["space_complexity"] = space_complexity;
//                return "You have provided the space complexity: " + space_complexity;
//            }

//            context.PromptInput();
//            return await this._chat.GenerateMessageAsync(chat, this._chatRequestSettings).ConfigureAwait(false);
//        }
//    }

//    public sealed class GiveFeedbackPlugin
//    {
//        private const string Goal = "As the coding interviewer, give user feedback and evaluate user's interview performance.";

//        private readonly IChatCompletion _chat;

//        private int MaxTokens { get; set; } = 256;

//        private readonly AIRequestSettings _chatRequestSettings;

//        public GiveFeedbackPlugin(IKernel kernel)
//        {
//            this._chat = kernel.GetService<IChatCompletion>();
//            this._chatRequestSettings = new OpenAIRequestSettings
//            {
//                MaxTokens = this.MaxTokens,
//                StopSequences = new List<string>() { "Observation:" },
//                Temperature = 0
//            };
//        }

//        [SKFunction]
//        [Description("This function is used to evaluate user's interview performance")]
//        [SKName("GiveFeedback")]
//        public async Task<string> GiveFeedbackAsync(
//            [SKName("solution")][Description("The full implementation given by the user")] string solution,
//            [SKName("time_complexity")][Description("The time complexity of the user's final solution")] string time_complexity,
//            [SKName("score")][Description("score for user's interview performance")] string score,
//            SKContext context)
//        {
//            Console.WriteLine("<======= Creating GiveFeedbackPlugin chat =======>\n");
//            var SystemPrompt =
//                "Based on user's coding interview solution: " + solution +
//                " and time complexity " + time_complexity +
//                "Score user's performance. Give user 4 scores out of 10: " +
//                "code reliabiity, code readability, communication score, test case walk-through. " +
//                "Tally up total scores out of 40. Tell user they passed the interview if above 32. " +
//                "Otherwise, tell them they failed and can retake after a month.";

//            var chat = this._chat.CreateNewChat(SystemPrompt);
//            chat.AddUserMessage(Goal);

//            ChatHistory? chatHistory = context.GetChatHistory();
//            if (chatHistory?.Any() ?? false)
//            {
//                chat.Messages.AddRange(chatHistory);
//            }

//            if (!string.IsNullOrEmpty(score))
//            {
//                Console.WriteLine($"Got Score: {score}");
//                context.Variables["score"] = score;
//                return score;
//            }

//            return await this._chat.GenerateMessageAsync(chat, this._chatRequestSettings).ConfigureAwait(false);
//        }
//    }
//}








//// Copyright (c) Microsoft. All rights reserved.

//using System;
//using System.Collections.Generic;
//using System.ComponentModel;
//using System.Diagnostics;
//using System.Linq;
//using System.Text;
//using System.Text.Json;
//using System.Text.RegularExpressions;
//using System.Threading.Tasks;
//using Azure.Search.Documents.Models;
//using Google.Apis.CustomSearchAPI.v1.Data;
//using Kusto.Data.Common;
//using Microsoft.Extensions.Logging;
//using Microsoft.SemanticKernel;
//using Microsoft.SemanticKernel.AI;
//using Microsoft.SemanticKernel.AI.ChatCompletion;
//using Microsoft.SemanticKernel.Connectors.AI.OpenAI;
//using Microsoft.SemanticKernel.Experimental.Orchestration;
//using Microsoft.SemanticKernel.Orchestration;
//using Microsoft.SemanticKernel.Plugins.Core;
//using Microsoft.SemanticKernel.Plugins.Memory;
//using Microsoft.SemanticKernel.Plugins.Web;
//using Microsoft.SemanticKernel.Plugins.Web.Bing;

///**
// * This example shows how to use FlowOrchestrator to execute a given flow with interaction with client.
// */

//// ReSharper disable once InconsistentNaming
//public static class Example62_FlowOrchestrator
//{
//    private static readonly Flow s_flow = FlowSerializer.DeserializeFromYaml(@"
//name: FlowOrchestrator_Example_Flow
//goal: generate interview problem and conduct coding interview for user
//steps:
//  - goal: As a coding interviewer, generate a coding interview problem that falls under the category of Dynamic Programming for the user to solve.
//    plugins:
//        - GenerateProblemPlugin
//    provides:
//        - problem_statement
//  - goal: Prompt user for questions related to logistics about the interview process. Ask user to code the problem solution.
//    plugins:
//        - PromptUserPlugin
//        - PromptSolutionPlugin
//    requires:
//        - problem_statement
//    completionType: AtLeastOnce
//    transitionMessage: do you have a better solution?
//    provides:
//      - user_questions
//      - solution
//      - time_complexity
//");
//    public static Task RunAsync()
//    {
//        return RunExampleAsync();
//    }

//    private static async Task RunExampleAsync()
//    {
//        var bingConnector = new BingConnector(TestConfiguration.Bing.ApiKey);
//        var webSearchEnginePlugin = new WebSearchEnginePlugin(bingConnector);
//        using var loggerFactory = LoggerFactory.Create(loggerBuilder =>
//            loggerBuilder
//                .AddConsole()
//                .AddFilter(null, LogLevel.Warning));
//        Dictionary<object, string?> plugins = new()
//        {
//            { webSearchEnginePlugin, "WebSearch" },
//            { new TimePlugin(), "time" }
//        };

//        FlowOrchestrator orchestrator = new(
//            GetKernelBuilder(loggerFactory),
//            await FlowStatusProvider.ConnectAsync(new VolatileMemoryStore()).ConfigureAwait(false),
//            plugins,
//            config: GetOrchestratorConfig());
//        var sessionId = Guid.NewGuid().ToString();

//        Console.WriteLine("*****************************************************");
//        Stopwatch sw = new();
//        sw.Start();
//        Console.WriteLine("Flow: " + s_flow.Name);
//        var result = await orchestrator.ExecuteFlowAsync(s_flow, sessionId, "Execute the flow").ConfigureAwait(false);

//        Console.WriteLine("Interviewer: " + result.ToString());

//        string[] userInputs = new[]
//        {
//            "Are there any constraints on the size of the input?",
//            "Can I assume the input will always be valid?",
//            "Do I need to implement the whole solution or just pseudo code is enough?",
//            "A brute force solution I can think of is to loop through all possible subarrays of the input list.",
//            "Well, I also observe that the maximum sum of a contiguous subarray ending at index i depends on the maximum sum of a contiguous subarray ending at index i-1.",
//            "I will initialize max_sum and current_sum to the first element of the list. Then, I will loop through the list, adding each element to current_sum. If current_sum is greater than max_sum, I will update max_sum. If current_sum is negative, I will reset it to zero. After processing all elements, I will return max_sum, which represents the maximum sum of any contiguous subarray within the list.",
//            "def max_subarray_sum(arr):\n    n = len(arr)\n    max_sum = arr[0]\n    current_sum = arr[0]\n    for i in range(1, n):\n        current_sum = max(arr[i], current_sum + arr[i])\n        max_sum = max(max_sum, current_sum)\n    return max_sum",
//            "I think the time complexity is O(n) since I iterate through the array exactly once.",
//        };

//        foreach (var t in userInputs)
//        {
//            Console.WriteLine($"\nUser: {t}");
//            result = await orchestrator.ExecuteFlowAsync(s_flow, sessionId, t).ConfigureAwait(false);
//            Console.WriteLine("Interviewer: " + result.ToString());
//        }

//        Console.WriteLine("Time Taken: " + sw.Elapsed);
//        Console.WriteLine("*****************************************************");
//    }
//    private static FlowOrchestratorConfig GetOrchestratorConfig()
//    {
//        var config = new FlowOrchestratorConfig
//        {
//            ReActModel = FlowOrchestratorConfig.ModelName.GPT35_TURBO,
//            MaxStepIterations = 20
//        };

//        return config;
//    }

//    private static KernelBuilder GetKernelBuilder(ILoggerFactory loggerFactory)
//    {
//        var builder = new KernelBuilder();

//        return builder
//            .WithAzureChatCompletionService(
//                TestConfiguration.AzureOpenAI.ChatDeploymentName,
//                TestConfiguration.AzureOpenAI.Endpoint,
//                TestConfiguration.AzureOpenAI.ApiKey,
//                true,
//                setAsDefault: true)
//            .WithRetryBasic(new()
//            {
//                MaxRetryCount = 3,
//                UseExponentialBackoff = true,
//                MinRetryDelay = TimeSpan.FromSeconds(3),
//            })
//                .WithLoggerFactory(
//                    LoggerFactory.Create(option => option.AddConsole()));
//    }

//    public sealed class AskUserToSeeIfTheyAreReadyToStartTheInterviewPlugin
//    {
//        private const string Goal = "As the coding interviewer, ask if user is ready to start the interview";

//        private const string SystemPrompt =
//            "Ask if user is ready to start the interview";

//        private readonly IChatCompletion _chat;

//        private int MaxTokens { get; set; } = 256;

//        private readonly AIRequestSettings _chatRequestSettings;

//        public AskUserToSeeIfTheyAreReadyToStartTheInterviewPlugin(IKernel kernel)
//        {
//            this._chat = kernel.GetService<IChatCompletion>();
//            this._chatRequestSettings = new OpenAIRequestSettings
//            {
//                MaxTokens = this.MaxTokens,
//                StopSequences = new List<string>() { "Observation:" },
//                Temperature = 0
//            };
//        }

//        [SKFunction]
//        [Description("This function is used to tell user rules for the interview process")]
//        [SKName("AskUserToSeeIfTheyAreReadyToStartTheInterviewPlugin")]
//        public async Task<string> AskUserToSeeIfTheyAreReadyToStartTheInterviewAsync(
//            //[SKName("problem_statement")][Description("The coding problem generated by the assistant")] string problem,
//            //[SKName("user_questions")][Description("User's questions")] string questions,
//            //[SKName("assistant_answers")][Description("Assistant's answers")] string answers,
//            SKContext context)
//        {
//            Console.WriteLine("<======= Creating AskUserToSeeIfTheyAreReadyToStartTheInterviewPlugin chat =======>\n");
//            var chat = this._chat.CreateNewChat(SystemPrompt);
//            chat.AddUserMessage(Goal);

//            //Console.WriteLine("Recall stated problem is: \n" + context.Variables["problem_statement"] + "\n");

//            ChatHistory? chatHistory = context.GetChatHistory();
//            if (chatHistory?.Any() ?? false)
//            {
//                chat.Messages.AddRange(chatHistory);
//            }

//            context.PromptInput();

//            //Console.WriteLine("After if clarifications not null\n" + chatHistory.ToString() + '\n');

//            return await this._chat.GenerateMessageAsync(chat, this._chatRequestSettings).ConfigureAwait(false);
//        }
//    }

//    public sealed class GenerateProblemPlugin
//    {
//        private const string Goal = "Generate a coding interview problem as an action.";

//        private const string SystemPrompt =
//            "You are an online coding interviewer. " +
//            "You will greet the candidate then " +
//            "generate a dynamic programming problem for the user to solve " +
//            "without telling them to use dynamic programming to solve the problem. " +
//            "Do not solve the problem for the user!";

//        private readonly IChatCompletion _chat;

//        private int MaxTokens { get; set; } = 256;

//        private readonly AIRequestSettings _chatRequestSettings;

//        public GenerateProblemPlugin(IKernel kernel)
//        {
//            this._chat = kernel.GetService<IChatCompletion>();
//            this._chatRequestSettings = new OpenAIRequestSettings
//            {
//                MaxTokens = this.MaxTokens,
//                StopSequences = new List<string>() { "Observation:" },
//                Temperature = 0
//            };
//        }

//        [SKFunction]
//        [Description("This function is used to generate a coding problem for user to solve")]
//        [SKName("GenerateProblem")]
//        public async Task<string> GenerateProblemAsync(
//            [SKName("problem_statement")][Description("The coding problem generated by the assistant")] string problem,
//            SKContext context)
//        {
//            Console.WriteLine("<======= Creating GenerateProblem chat =======>\n");
//            var chat = this._chat.CreateNewChat(SystemPrompt);
//            chat.AddUserMessage(Goal);

//            ChatHistory? chatHistory = context.GetChatHistory();

//            if (chatHistory?.Any() ?? false)
//            {
//                chat.Messages.AddRange(chatHistory);
//            }
//            if (!string.IsNullOrEmpty(problem))
//            {
//                context.Variables["problem_statement"] = problem;

//                return problem;
//            }
//            context.Variables["problem_statement"] = string.Empty;
//            var answer = await this._chat.GenerateMessageAsync(chat, this._chatRequestSettings).ConfigureAwait(false);
//            Console.WriteLine("Interviewer: " + answer + '\n');

//            return answer;
//        }
//    }

//    public sealed class PromptUserPlugin
//    {
//        private const string Goal = "You have generated a coding problem. Now answer user's questions until user is ready to solve the problem.";

//        private const string SystemPrompt =
//            "You are an online coding interviewer. You have provided a coding problem for the user to solve. " +
//            "If the user asks a question regarding logistics of the interview, answer them. " +
//            "Otherwise, do not provide extra information the user does not ask for. " +
//            "For example, you can answer questions like the what is the interview process like " +
//            "can I use any language to write the solution " +
//            "or how big is the input " +
//            "but not questions directly related to solving the problem. " +
//            "Ask if user has any more questions.";
//        private readonly IChatCompletion _chat;

//        private int MaxTokens { get; set; } = 256;

//        private readonly AIRequestSettings _chatRequestSettings;

//        public PromptUserPlugin(IKernel kernel)
//        {
//            this._chat = kernel.GetService<IChatCompletion>();
//            this._chatRequestSettings = new OpenAIRequestSettings
//            {
//                MaxTokens = this.MaxTokens,
//                StopSequences = new List<string>() { "Observation:" },
//                Temperature = 0
//            };
//        }

//        [SKFunction]
//        [Description("This function is used to prompt user for questions related to interview logistics and answer them")]
//        [SKName("PromptUserPlugin")]
//        public async Task<string> PromptUserPluginAsync(
//            [SKName("ready")][Description("User's response to if user has more questions.")] string ready,
//            [SKName("problem_statement")][Description("The coding problem generated by the assistant")] string problem,
//            //[SKName("user_questions")][Description("User's questions")] string questions,
//            //[SKName("assistant_answers")][Description("Assistant's answers")] string answers,
//            SKContext context)
//        {
//            Console.WriteLine("<======= Creating PromptUserPlugin chat =======>\n");
//            if (!string.IsNullOrEmpty(ready) && ready == "yes")
//            {
//                context.Variables["ready"] = "yes";

//                return "Glad we got everything clarified!";
//            }
//            var chat = this._chat.CreateNewChat(SystemPrompt);
//            chat.AddUserMessage(Goal);

//            //Console.WriteLine("Recall stated problem is: \n" + context.Variables["problem_statement"] + "\n");

//            ChatHistory? chatHistory = context.GetChatHistory();
//            if (chatHistory?.Any() ?? false)
//            {
//                chat.Messages.AddRange(chatHistory);
//            }

//            context.PromptInput();

//            return await this._chat.GenerateMessageAsync(chat, this._chatRequestSettings).ConfigureAwait(false);
//        }
//    }
//    public sealed class PromptSolutionPlugin
//    {
//        private const string Goal = "Ask user to solve the problem and get the solution from the user";

//        private const string SystemPrompt =
//            "As the user dissects how to solve the problem, give hints if the user is stuck; " +
//            "otherwise let the user to continue analyzing the problem. " +
//            "After the user comes up with pseudo code, " +
//            "ask user to explain the time complexity and prompt user to implement the solution. " +
//            "If the full implementation given by the user isn't the optimal solution, " +
//            "ask if the user has a better solution to improve the time complexity. " +
//            "Ask user if they have given you their final solution.";

//        private readonly IChatCompletion _chat;

//        private int MaxTokens { get; set; } = 256;

//        private readonly AIRequestSettings _chatRequestSettings;

//        public PromptSolutionPlugin(IKernel kernel)
//        {
//            this._chat = kernel.GetService<IChatCompletion>();
//            this._chatRequestSettings = new OpenAIRequestSettings
//            {
//                MaxTokens = this.MaxTokens,
//                StopSequences = new List<string>() { "Observation:" },
//                Temperature = 0
//            };
//        }

//        [SKFunction]
//        [Description("This function is used to get from user the solution to the coding problem")]
//        [SKName("PromptSolution")]
//        public async Task<string> PromptSolutionAsync(
//            [SKName("solution")][Description("The final solution given by the user")] string solution,
//            [SKName("time_complexity")][Description("The time complexity of the user's final solution")] string time_complexity,
//            SKContext context)
//        {
//            Console.WriteLine("<======= Creating PromptSolution chat =======>\n");
//            var chat = this._chat.CreateNewChat(SystemPrompt);
//            chat.AddUserMessage(Goal);

//            ChatHistory? chatHistory = context.GetChatHistory();
//            if (chatHistory?.Any() ?? false)
//            {
//                chat.Messages.AddRange(chatHistory);
//            }

//            if (!string.IsNullOrEmpty(solution))
//            {
//                context.Variables["solution"] = solution;
//                if (!string.IsNullOrEmpty(time_complexity))
//                {
//                    context.Variables["time_complexity"] = time_complexity;

//                    return "You have provided the solution: " + solution +
//                           " and your time complexity is: " + time_complexity;
//                }

//                return "You have provided the solution: " + solution;
//            }

//            context.Variables["solution"] = solution ?? string.Empty;
//            context.Variables["time_complexity"] = string.Empty;

//            context.PromptInput();

//            return await this._chat.GenerateMessageAsync(chat, this._chatRequestSettings).ConfigureAwait(false);
//        }
//    }
//}
