// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.AI;
using Microsoft.SemanticKernel.AI.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.AI.OpenAI;
using Microsoft.SemanticKernel.Experimental.Orchestration;
using Microsoft.SemanticKernel.Orchestration;

namespace Interviewer;
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
