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
        Console.WriteLine("<======= Creating GenerateProblem chat =======>\n");
        var chat = this._chat.CreateNewChat(SystemPrompt);
        chat.AddUserMessage(Goal);

        ChatHistory? chatHistory = context.GetChatHistory();
        if (chatHistory?.Any() ?? false)
        {
            chat.Messages.AddRange(chatHistory);
        }

        Console.Write("000000000000");

        if (!string.IsNullOrEmpty(problem))
        {
            context.Variables["problem_statement"] = problem;
            Console.WriteLine("Assistant: Hello! Thanks for joining the coding interview. " +
               "Here's the problem for you to solve: \n" + problem);

            return "Assistant: Hello! Thanks for joining the coding interview. " +
               "Here's the problem for you to solve: \n" + problem;
        }

        return "Assistant: " + await this._chat.GenerateMessageAsync(chat, this._chatRequestSettings).ConfigureAwait(false);
    }
}
